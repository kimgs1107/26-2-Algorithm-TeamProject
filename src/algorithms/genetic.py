"""
Memetic Genetic Algorithm (GA + Domain Local Search) for optimal card assignment.

- Chromosome: integer array [c_0, c_1, ..., c_{n-1}] assigning each chronological
  transaction of a month to a card index in range(len(cards)).
- Exact Fitness: matches common.evaluate_assignment 100% down to 1 Won, reflecting
  chronological cumulative spend gates (spend[card] >= min_spend) and monthly
  per-(card, category) benefit caps via vectorized BLAS matrix operations.
- Evolutionary Operators:
    1) Diverse initialization: uniform random + raw-benefit weighted random +
       chronological gate-subset & remaining-cap-aware stochastic seeds.
    2) Tournament selection (k=3) + elitism.
    3) Hybrid crossover: chronological single-point split, two-point, and uniform.
    4) Adaptive sparse point mutation (p_m ~ 2/n) + gate-boost/evacuate &
       post-cap spillover macro mutation.
    5) Memetic local refinement: exact incremental O(1)/O(log n) 1-opt coordinate
       polish, zero-cost gate-cushion & early-unlock routing, and targeted 2-opt
       swaps on elite individuals.
"""

from __future__ import annotations

import itertools

import numpy as np

try:
    from .common import (
        build_benefit_table,
        clean_transactions,
        evaluate_assignment,
        format_result,
        split_by_month,
    )
except ImportError:
    try:
        from common import (
            build_benefit_table,
            clean_transactions,
            evaluate_assignment,
            format_result,
            split_by_month,
        )
    except ImportError:
        import sys as _sys
        from pathlib import Path as _Path

        _sys.path.append(str(_Path(__file__).resolve().parent))
        from common import (
            build_benefit_table,
            clean_transactions,
            evaluate_assignment,
            format_result,
            split_by_month,
        )

INF_LIMIT = 10**15


# =====================================================================
# 1. Month Benefit Model & Vectorized Population Evaluator
# =====================================================================


def _raw_candidate_benefit(candidate: dict, tx_amount: int) -> int:
    """Compute uncapped single-transaction benefit for one candidate entry."""
    if candidate["rate"] is not None:   # 정률 할인 -> 결제액 * 할인율
        return int(tx_amount * candidate["rate"])
    if candidate["flat"] is not None:   # 정액 할인 -> 할인액과 결제액 중 작은 값 반환
        return min(int(candidate["flat"]), tx_amount)
    return 0


def _build_month_model(month_txs: list[dict], cards: list[str], table: dict) -> dict:
    """
    Precompute card/bucket matrices and a fast vectorized population evaluator
    for a single month's chronologically sorted transactions.
    """
    num_txs = len(month_txs)    # 결제 건수
    num_cards = len(cards)      # 보유 카드 개수
    amounts_i64 = np.array([int(tx["금액"]) for tx in month_txs], dtype=np.int64)   # 금액만 추출하여 배열화
    total_spend = int(amounts_i64.sum())    # 월 총 결제 금액

    raw_elig = np.zeros((num_txs, num_cards), dtype=np.int64)   # 카드 실적 채웠을 때 얻는 기본 혜택액
    raw_inelig = np.zeros((num_txs, num_cards), dtype=np.int64) # 카드 실적 채우지 못했을 때 얻는 기본 혜택액
    card_gates = np.zeros(num_cards, dtype=np.int64)    # 각 카드 전월 실적 기준
    card_has_unlimited = np.zeros(num_cards, dtype=bool)    # 한도 제한 없는 혜택 있는지(bool)
    card_specs = [] # 카드별 전처리 dictionary 저장 list

    for card_idx, card_id in enumerate(cards):  # card_idx, card_id 순회
        gate = int(table["gates"].get(card_id, 0))  # 카드별 전월 실적 기준
        card_gates[card_idx] = gate # 해당하는 위치에 해당 카드 전월 실적 기준 저장
        cat_benefits = table["benefits"][card_id]   # 해당 카드 카테고리별 혜택 정보 추출

        for cat_cands in cat_benefits.values(): # cat_benefits 내 각 값 순회
            for cand in cat_cands:  # 각 값 내 개별 혜택 확인
                if table["limits"].get(cand["limit_key"]) is None:  # 한도 제한 없는 혜택 있는 경우 card_has_unlimited 내에 기록하고 탈출
                    card_has_unlimited[card_idx] = True
                    break

        active_cats = []    # 최종 결과 담을 list
        seen_cats = set()   # 확인한 업종 따로 set에 저장(중복 탐색 방지용)
        for tx in month_txs:    # 이번 달 거래 내역 순회
            cat = tx["업종"]    # 업종명
            if cat not in seen_cats and cat_benefits.get(cat):  # 아직 목록에 추가하지 않음 & 혜택 정보 존재 + 유횻값 -> 확인한 업종 set, 최종 결과 list에 추가
                seen_cats.add(cat)
                active_cats.append(cat)

        cat_to_col = {cat: idx for idx, cat in enumerate(active_cats)}  # 업종명 문자열을 열 번호에 mapping
        num_buckets = len(active_cats)  # active_cats 전체 카테고리 개수 

        bucket_limits_i64 = np.empty(num_buckets, dtype=np.int64)   # 정수형 배열
        bucket_limits_f64 = np.empty(num_buckets, dtype=np.float64) # 실수형 배열
        for idx, cat in enumerate(active_cats): # index, 카테고리 순서대로 꺼냄
            lim = table["limits"].get(f"{card_id}:{cat}")   # 해당 카드, 카테고리 한계
            if lim is None: # 한도 없는 경우: 정수 배열에 INF_LIMIT, 실수 배열에 무한대 삽입
                bucket_limits_i64[idx] = INF_LIMIT
                bucket_limits_f64[idx] = np.inf
            else:   # 한도 있는 경우: 두 배열에 모두 한돗값 저장
                bucket_limits_i64[idx] = int(lim)
                bucket_limits_f64[idx] = float(lim)

        mat_elig = np.zeros((num_txs, num_buckets), dtype=np.float64)   # 실적 충족 시 혜택(eligible) 계산에 사용할 행렬
        mat_inelig = np.zeros((num_txs, num_buckets), dtype=np.float64) # 실적 충족 불필요한 혜택(ineligible) 계산에 사용할 행렬
        tx_bucket = np.full(num_txs, -1, dtype=np.int32)    # 각 거래가 어느 업종에 속하는지 기록(-1로 초기화)
        has_inelig = False  # ineligible 여부 확인 boolean flag

        for tx_idx, tx in enumerate(month_txs): # 각 결제마다 해당 업종 혜택 후보 순회
            col = cat_to_col.get(tx["업종"], -1)    # 결제 업종이 분석 대상 카테고리인지 확인, 맞으면 열 번호 저장(아니면 -1)
            if col < 0: # 분석 대상 카테고리 아니면 건너뛰기
                continue
            tx_bucket[tx_idx] = col #tx_bucket에 col 값 저장
            amt = int(tx["금액"])   # 결제액 integer 자료형 변환
            best_elig = 0   # 실적 충족 시 최대 혜택
            best_inelig = 0 # 실적 충족 불필요한 최대 혜택
            for cand in cat_benefits[tx["업종"]]:   # 적용 가능한 모든 혜택 순회하며 개별 혜택 금액 계산
                val = _raw_candidate_benefit(cand, amt)
                if val > best_elig: # [best_elig] 계산한 값이 기존 최댓값보다 크면 덮어쓰기
                    best_elig = val
                if cand["min_spend"] <= 0 and val > best_inelig:    # [best_inelig] 계산한 값이 기존 최댓값보다 크고 실적 <= 0일 경우 덮어쓰기
                    best_inelig = val
            raw_elig[tx_idx, card_idx] = best_elig  # raw_elig 2차원 배열에 최댓값 저장
            raw_inelig[tx_idx, card_idx] = best_inelig  #raw_inelig 2차원 배열에 최댓값 저장
            mat_elig[tx_idx, col] = float(best_elig)    # raw_elig 2차원 배열에 float 자료형으로 최댓값 저장
            mat_inelig[tx_idx, col] = float(best_inelig)    # raw_inelig 2차원 배열에 float 자료형으로 최댓값 저장
            if best_inelig > 0: # 무실적 혜택 하나라도 있으면 true로 저장
                has_inelig = True

        card_specs.append(  # 생성한 카드별 데이터 dictionary로 묶어 추가
            {
                "card_idx": card_idx,
                "gate": gate,
                "num_buckets": num_buckets,
                "bucket_limits_i64": bucket_limits_i64,
                "bucket_limits_f64": bucket_limits_f64,
                "mat_elig": mat_elig,
                "mat_inelig": mat_inelig,
                "mat_delta": mat_elig - mat_inelig,
                "has_inelig": has_inelig,
                "tx_bucket": tx_bucket,
            }
        )

    def evaluate_population(pop: np.ndarray) -> np.ndarray:
        pop_len = pop.shape[0]  # 카드 수
        totals = np.zeros(pop_len, dtype=np.int64)  # 총 혜택액
        for spec in card_specs: # 카드별 순회
            if spec["num_buckets"] == 0:    # 혜택 없으면 건너뜀
                continue
            c_idx = spec["card_idx"]    # 현재 순회 중인 카드 index
            mask = pop == c_idx # 전체 배열에서 현재 카드가 선택된 위치만 masking
            gate = spec["gate"] # 혜택 받기 위한 최소 조건 data
            if gate <= 0:   # 실적 조건 없는 경우
                raw_sums = mask.astype(np.float64) @ spec["mat_elig"]   # 모든 결제 건이 혜택 대상 -> mask와 각 결제 건 혜택 행렬 행렬곱 수행
            else:
                spend_steps = np.where(mask, amounts_i64[None, :], 0)
                cum_spend = np.cumsum(spend_steps, axis=1)
                elig_mask = mask & (cum_spend >= gate)
                if spec["has_inelig"]:
                    raw_sums = (
                        mask.astype(np.float64) @ spec["mat_inelig"]
                        + elig_mask.astype(np.float64) @ spec["mat_delta"]
                    )
                else:
                    raw_sums = elig_mask.astype(np.float64) @ spec["mat_elig"]
            capped = np.minimum(raw_sums, spec["bucket_limits_f64"])
            totals += capped.sum(axis=1).astype(np.int64)
        return totals

    return {
        "num_txs": num_txs,
        "num_cards": num_cards,
        "amounts_i64": amounts_i64,
        "total_spend": total_spend,
        "raw_elig": raw_elig,
        "raw_inelig": raw_inelig,
        "card_gates": card_gates,
        "card_has_unlimited": card_has_unlimited,
        "card_specs": card_specs,
        "evaluate_population": evaluate_population,
    }


# =====================================================================
# 2. Exact Incremental State for Memetic Local Refinement
# =====================================================================


class _IncrementalMonthState:
    """
    Maintains exact per-card cumulative spend, gate-crossing index, and category
    bucket sums so single-gene 1-opt moves and swaps evaluate in O(1)/O(log n).
    """

    def __init__(self, chrom: np.ndarray, model: dict) -> None:
        self.chrom = chrom.astype(np.int16, copy=True)
        self.amounts = model["amounts_i64"]
        self.raw_elig = model["raw_elig"]
        self.raw_inelig = model["raw_inelig"]
        self.specs = model["card_specs"]
        self.num_txs = int(model["num_txs"])
        self.num_cards = int(model["num_cards"])

        self.bucket_sums: list[np.ndarray] = []
        self.card_scores = np.zeros(self.num_cards, dtype=np.int64)
        self.cum_spends: list[np.ndarray | None] = []
        self.total_spends = np.zeros(self.num_cards, dtype=np.int64)
        self.cross_indices = np.full(self.num_cards, self.num_txs, dtype=np.int64)

        for c_idx, spec in enumerate(self.specs):
            nb = spec["num_buckets"]
            bsum = np.zeros(nb, dtype=np.int64)
            mask = self.chrom == c_idx
            gate = spec["gate"]
            if gate > 0:
                sp_steps = np.where(mask, self.amounts, 0)
                cs = np.cumsum(sp_steps, dtype=np.int64)
                tot_sp = int(cs[-1]) if self.num_txs > 0 else 0
                c_cross = (
                    int(np.searchsorted(cs, gate, side="left"))
                    if tot_sp >= gate
                    else self.num_txs
                )
                self.cum_spends.append(cs)
                self.total_spends[c_idx] = tot_sp
                self.cross_indices[c_idx] = c_cross
            else:
                self.cum_spends.append(None)
                self.total_spends[c_idx] = int(self.amounts[mask].sum())
                self.cross_indices[c_idx] = 0

            if nb > 0:
                tx_b = spec["tx_bucket"]
                c_cross = int(self.cross_indices[c_idx])
                for ti in np.flatnonzero(mask):
                    b = int(tx_b[ti])
                    if b >= 0:
                        val = (
                            self.raw_elig[ti, c_idx]
                            if ti >= c_cross
                            else self.raw_inelig[ti, c_idx]
                        )
                        bsum[b] += val
                self.card_scores[c_idx] = int(
                    np.minimum(bsum, spec["bucket_limits_i64"]).sum()
                )
            self.bucket_sums.append(bsum)

        self.total_score = int(self.card_scores.sum())

    def delta_remove(self, c_idx: int, tx_idx: int) -> int:
        spec = self.specs[c_idx]
        if spec["num_buckets"] == 0:
            return 0
        gate = spec["gate"]
        tx_b = spec["tx_bucket"]
        bsum = self.bucket_sums[c_idx]
        lims = spec["bucket_limits_i64"]
        b_i = int(tx_b[tx_idx])

        if gate <= 0:
            if b_i < 0:
                return 0
            r = int(self.raw_elig[tx_idx, c_idx])
            if r == 0:
                return 0
            lim = int(lims[b_i])
            cur = int(bsum[b_i])
            return min(lim, cur - r) - min(lim, cur)

        tot_sp = int(self.total_spends[c_idx])
        old_cross = int(self.cross_indices[c_idx])
        amt = int(self.amounts[tx_idx])

        if tot_sp < gate:
            if b_i < 0:
                return 0
            r = int(self.raw_inelig[tx_idx, c_idx])
            if r == 0:
                return 0
            lim = int(lims[b_i])
            cur = int(bsum[b_i])
            return min(lim, cur - r) - min(lim, cur)

        if tx_idx > old_cross:
            if b_i < 0:
                return 0
            r = int(self.raw_elig[tx_idx, c_idx])
            if r == 0:
                return 0
            lim = int(lims[b_i])
            cur = int(bsum[b_i])
            return min(lim, cur - r) - min(lim, cur)

        cs = self.cum_spends[c_idx]
        assert cs is not None
        if tot_sp - amt < gate:
            new_cross = self.num_txs
        else:
            new_cross = old_cross + int(
                np.searchsorted(cs[old_cross:], gate + amt, side="left")
            )

        temp_bsum = bsum.copy()
        if b_i >= 0:
            val_i = (
                int(self.raw_elig[tx_idx, c_idx])
                if tx_idx >= old_cross
                else int(self.raw_inelig[tx_idx, c_idx])
            )
            temp_bsum[b_i] -= val_i

        if new_cross > old_cross:
            chrom_slice = self.chrom[old_cross:new_cross]
            for rel in np.flatnonzero(chrom_slice == c_idx):
                t = old_cross + int(rel)
                if t == tx_idx:
                    continue
                bt = int(tx_b[t])
                if bt >= 0:
                    temp_bsum[bt] -= int(
                        self.raw_elig[t, c_idx] - self.raw_inelig[t, c_idx]
                    )

        new_score = int(np.minimum(temp_bsum, lims).sum())
        return new_score - int(self.card_scores[c_idx])

    def delta_add(self, c_idx: int, tx_idx: int) -> int:
        spec = self.specs[c_idx]
        if spec["num_buckets"] == 0:
            return 0
        gate = spec["gate"]
        tx_b = spec["tx_bucket"]
        bsum = self.bucket_sums[c_idx]
        lims = spec["bucket_limits_i64"]
        b_i = int(tx_b[tx_idx])

        if gate <= 0:
            if b_i < 0:
                return 0
            r = int(self.raw_elig[tx_idx, c_idx])
            if r == 0:
                return 0
            lim = int(lims[b_i])
            cur = int(bsum[b_i])
            return min(lim, cur + r) - min(lim, cur)

        tot_sp = int(self.total_spends[c_idx])
        old_cross = int(self.cross_indices[c_idx])
        amt = int(self.amounts[tx_idx])

        if tot_sp + amt < gate:
            if b_i < 0:
                return 0
            r = int(self.raw_inelig[tx_idx, c_idx])
            if r == 0:
                return 0
            lim = int(lims[b_i])
            cur = int(bsum[b_i])
            return min(lim, cur + r) - min(lim, cur)

        if tx_idx > old_cross:
            if b_i < 0:
                return 0
            r = int(self.raw_elig[tx_idx, c_idx])
            if r == 0:
                return 0
            lim = int(lims[b_i])
            cur = int(bsum[b_i])
            return min(lim, cur + r) - min(lim, cur)

        cs = self.cum_spends[c_idx]
        assert cs is not None
        end_search = min(old_cross, self.num_txs)
        if cs[tx_idx] + amt >= gate:
            new_cross = tx_idx
        else:
            new_cross = tx_idx + int(
                np.searchsorted(cs[tx_idx:end_search], gate - amt, side="left")
            )

        temp_bsum = bsum.copy()
        if b_i >= 0:
            val_i = (
                int(self.raw_elig[tx_idx, c_idx])
                if tx_idx >= new_cross
                else int(self.raw_inelig[tx_idx, c_idx])
            )
            temp_bsum[b_i] += val_i

        if end_search > new_cross:
            chrom_slice = self.chrom[new_cross:end_search]
            for rel in np.flatnonzero(chrom_slice == c_idx):
                t = new_cross + int(rel)
                bt = int(tx_b[t])
                if bt >= 0:
                    temp_bsum[bt] += int(
                        self.raw_elig[t, c_idx] - self.raw_inelig[t, c_idx]
                    )

        new_score = int(np.minimum(temp_bsum, lims).sum())
        return new_score - int(self.card_scores[c_idx])

    def apply_move(self, tx_idx: int, new_card: int) -> None:
        old_card = int(self.chrom[tx_idx])
        if old_card == new_card:
            return
        amt = int(self.amounts[tx_idx])

        spec_old = self.specs[old_card]
        gate_old = spec_old["gate"]
        old_cross = int(self.cross_indices[old_card])
        if gate_old > 0:
            cs_old = self.cum_spends[old_card]
            assert cs_old is not None
            cs_old[tx_idx:] -= amt
            new_tot_old = int(self.total_spends[old_card]) - amt
            self.total_spends[old_card] = new_tot_old
            if new_tot_old < gate_old:
                new_cross_old = self.num_txs
            elif tx_idx > old_cross:
                new_cross_old = old_cross
            else:
                new_cross_old = old_cross + int(
                    np.searchsorted(cs_old[old_cross:], gate_old, side="left")
                )
            self.cross_indices[old_card] = new_cross_old
        else:
            self.total_spends[old_card] -= amt
            new_cross_old = 0

        if spec_old["num_buckets"] > 0:
            bsum_old = self.bucket_sums[old_card]
            tx_b_old = spec_old["tx_bucket"]
            b_i = int(tx_b_old[tx_idx])
            if b_i >= 0:
                val_i = (
                    int(self.raw_elig[tx_idx, old_card])
                    if tx_idx >= old_cross
                    else int(self.raw_inelig[tx_idx, old_card])
                )
                bsum_old[b_i] -= val_i
            if gate_old > 0 and new_cross_old > old_cross:
                chrom_slice = self.chrom[old_cross:new_cross_old]
                for rel in np.flatnonzero(chrom_slice == old_card):
                    t = old_cross + int(rel)
                    if t == tx_idx:
                        continue
                    bt = int(tx_b_old[t])
                    if bt >= 0:
                        bsum_old[bt] -= int(
                            self.raw_elig[t, old_card] - self.raw_inelig[t, old_card]
                        )
            new_old_score = int(
                np.minimum(bsum_old, spec_old["bucket_limits_i64"]).sum()
            )
            self.total_score += new_old_score - int(self.card_scores[old_card])
            self.card_scores[old_card] = new_old_score

        self.chrom[tx_idx] = new_card

        spec_new = self.specs[new_card]
        gate_new = spec_new["gate"]
        prev_cross_new = int(self.cross_indices[new_card])
        if gate_new > 0:
            cs_new = self.cum_spends[new_card]
            assert cs_new is not None
            cs_new[tx_idx:] += amt
            new_tot_new = int(self.total_spends[new_card]) + amt
            self.total_spends[new_card] = new_tot_new
            if new_tot_new < gate_new:
                new_cross_new = self.num_txs
            elif tx_idx > prev_cross_new:
                new_cross_new = prev_cross_new
            else:
                end_s = min(prev_cross_new, self.num_txs)
                if cs_new[tx_idx] >= gate_new:
                    new_cross_new = tx_idx
                else:
                    new_cross_new = tx_idx + int(
                        np.searchsorted(cs_new[tx_idx:end_s], gate_new, side="left")
                    )
            self.cross_indices[new_card] = new_cross_new
        else:
            self.total_spends[new_card] += amt
            new_cross_new = 0

        if spec_new["num_buckets"] > 0:
            bsum_new = self.bucket_sums[new_card]
            tx_b_new = spec_new["tx_bucket"]
            b_i = int(tx_b_new[tx_idx])
            if b_i >= 0:
                val_i = (
                    int(self.raw_elig[tx_idx, new_card])
                    if tx_idx >= new_cross_new
                    else int(self.raw_inelig[tx_idx, new_card])
                )
                bsum_new[b_i] += val_i
            if gate_new > 0:
                end_s = min(prev_cross_new, self.num_txs)
                if end_s > new_cross_new:
                    chrom_slice = self.chrom[new_cross_new:end_s]
                    for rel in np.flatnonzero(chrom_slice == new_card):
                        t = new_cross_new + int(rel)
                        if t == tx_idx:
                            continue
                        bt = int(tx_b_new[t])
                        if bt >= 0:
                            bsum_new[bt] += int(
                                self.raw_elig[t, new_card]
                                - self.raw_inelig[t, new_card]
                            )
            new_new_score = int(
                np.minimum(bsum_new, spec_new["bucket_limits_i64"]).sum()
            )
            self.total_score += new_new_score - int(self.card_scores[new_card])
            self.card_scores[new_card] = new_new_score


def _memetic_polish(
    chrom: np.ndarray,
    model: dict,
    rng: np.random.Generator,
    max_rounds: int = 6,
    enable_swaps: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Domain-specific memetic local refinement on a single chromosome:
      Pass A: Route zero-marginal-cost transactions from non-gated / post-cap cards
              onto active gated cards to pull cross_idx earlier & build spend cushion.
      Pass B: Exact 1-opt coordinate ascent across all transactions and cards.
      Pass C: Targeted 2-opt swaps exchanging gate-locked post-cap/ineligible
              transactions with external spend-building or complementary transactions.
    """
    num_txs = model["num_txs"]
    num_cards = model["num_cards"]
    if num_txs == 0 or num_cards <= 1:
        eval_fn = model["evaluate_population"]
        score = int(eval_fn(chrom[None, :])[0]) if num_txs > 0 else 0
        return chrom.copy(), score

    state = _IncrementalMonthState(chrom, model)
    card_gates = model["card_gates"]
    gated_cards = [c for c in range(num_cards) if card_gates[c] > 0]

    for round_idx in range(max_rounds):
        improved_in_round = False

        # Pass A: Zero-loss gate cushion & early-unlock routing onto active gated cards
        if gated_cards:
            active_gated = [
                c for c in gated_cards if state.total_spends[c] >= card_gates[c]
            ]
            # Also include nearly-unlocked gated cards if within 35% of gate on round 0
            if round_idx == 0:
                near_gated = [
                    c
                    for c in gated_cards
                    if int(0.65 * card_gates[c]) <= state.total_spends[c] < card_gates[c]
                ]
                active_gated.extend(near_gated)

            if active_gated:
                # Prioritize gated cards that cross late (high cross_indices)
                active_gated.sort(key=lambda c: -int(state.cross_indices[c]))
                for ti in range(num_txs):
                    src = int(state.chrom[ti])
                    if src in active_gated:
                        continue
                    # Only move if removing from src neither lowers score nor weakens an active gate
                    if card_gates[src] > 0 and state.total_spends[src] >= card_gates[src]:
                        if state.total_spends[src] - state.amounts[ti] < card_gates[src]:
                            continue
                        if ti <= state.cross_indices[src]:
                            continue
                    rem_delta = state.delta_remove(src, ti)
                    if rem_delta < 0:
                        continue

                    # Pick active gated card that benefits most (or still needs early crossing)
                    best_tc = -1
                    best_gain = -1
                    for tc in active_gated:
                        add_delta = state.delta_add(tc, ti)
                        net = rem_delta + add_delta
                        if net > best_gain:
                            best_gain = net
                            best_tc = tc
                        elif net == 0 and best_gain <= 0 and ti < state.cross_indices[tc]:
                            best_gain = 0
                            best_tc = tc
                    if best_tc >= 0 and (
                        best_gain > 0
                        or (
                            best_gain == 0
                            and (
                                ti < state.cross_indices[best_tc]
                                or model["raw_elig"][ti, src] == 0
                            )
                        )
                    ):
                        prev_score = state.total_score
                        state.apply_move(ti, best_tc)
                        if state.total_score > prev_score:
                            improved_in_round = True

        # Pass B: 1-opt coordinate ascent sweep
        tx_order = rng.permutation(num_txs) if round_idx % 2 == 1 else np.arange(num_txs)
        for ti_raw in tx_order:
            ti = int(ti_raw)
            cur_c = int(state.chrom[ti])
            rem_delta = state.delta_remove(cur_c, ti)
            best_c = cur_c
            best_delta = 0
            for cand_c in range(num_cards):
                if cand_c == cur_c:
                    continue
                cand_delta = rem_delta + state.delta_add(cand_c, ti)
                if cand_delta > best_delta:
                    best_delta = cand_delta
                    best_c = cand_c
            if best_c != cur_c:
                state.apply_move(ti, best_c)
                improved_in_round = True

        # Pass C: Targeted 2-opt swaps for tight gate / cap interactions
        if enable_swaps and gated_cards and num_txs <= 1500:
            active_gated = [
                c for c in gated_cards if state.total_spends[c] >= card_gates[c]
            ]
            if active_gated:
                # Candidate transactions on active gated cards that currently yield 0 or capped gain
                gated_set = set(active_gated)
                on_gate_candidates = []
                off_gate_candidates = []
                for ti in range(num_txs):
                    ci = int(state.chrom[ti])
                    if ci in gated_set:
                        # Check if moving ti off ci loses less than potential elsewhere, or is gate-locked
                        max_alt = int(model["raw_elig"][ti].max())
                        if max_alt > 0:
                            on_gate_candidates.append(ti)
                    else:
                        off_gate_candidates.append(ti)

                if len(on_gate_candidates) > 40:
                    on_gate_candidates = rng.choice(
                        on_gate_candidates, size=40, replace=False
                    ).tolist()
                if len(off_gate_candidates) > 50:
                    off_gate_candidates = rng.choice(
                        off_gate_candidates, size=50, replace=False
                    ).tolist()

                for ti in on_gate_candidates:
                    ci = int(state.chrom[ti])
                    for tj in off_gate_candidates:
                        cj = int(state.chrom[tj])
                        if ci == cj:
                            continue
                        before = state.total_score
                        state.apply_move(ti, cj)
                        state.apply_move(tj, ci)
                        if state.total_score > before:
                            improved_in_round = True
                            ci = int(state.chrom[ti])
                            break
                        else:
                            # Revert swap cleanly
                            state.apply_move(tj, cj)
                            state.apply_move(ti, ci)

        if not improved_in_round:
            break

    return state.chrom.copy(), int(state.total_score)


# =====================================================================
# 3. Initial Population Construction & Evolutionary Engine
# =====================================================================


def _construct_subset_seed(
    model: dict,
    target_gates: list[int],
    urgency_weight: float,
    eps: float,
    preassign_zero_opp: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build one chronological seed chromosome targeting a feasible subset of gated
    cards while respecting remaining per-(card, category) benefit caps.
    """
    num_txs = model["num_txs"]
    num_cards = model["num_cards"]
    amounts = model["amounts_i64"]
    raw_elig = model["raw_elig"]
    raw_inelig = model["raw_inelig"]
    card_gates = model["card_gates"]
    specs = model["card_specs"]

    target_set = set(target_gates)
    active_cards = [
        c
        for c in range(num_cards)
        if (c in target_set)
        or (card_gates[c] == 0)
        or bool(np.any(raw_inelig[:, c] > 0))
    ]
    if not active_cards:
        active_cards = list(range(num_cards))

    chrom = np.full(num_txs, -1, dtype=np.int16)
    cur_spend = [0] * num_cards
    rem_caps = [spec["bucket_limits_i64"].copy() for spec in specs]

    # Optional pre-pass: route early zero-benefit ("기타") transactions directly to target gates
    if preassign_zero_opp and target_gates:
        gate_order = list(target_gates)
        rng.shuffle(gate_order)
        g_ptr = 0
        for ti in range(num_txs):
            if g_ptr >= len(gate_order):
                break
            if int(raw_elig[ti].max()) == 0:
                tc = gate_order[g_ptr]
                chrom[ti] = tc
                cur_spend[tc] += int(amounts[ti])
                if cur_spend[tc] >= int(card_gates[tc]):
                    g_ptr += 1

        # Rebuild cur_spend to run chronological pass cleanly
        cur_spend = [0] * num_cards

    rand_u = rng.random(num_txs)
    jitter = rng.uniform(0.0, 6.0, size=(num_txs, num_cards))
    inv_n = 1.0 / max(1, num_txs)

    for ti in range(num_txs):
        amt = int(amounts[ti])

        if chrom[ti] >= 0:
            picked = int(chrom[ti])
        elif rand_u[ti] < eps:
            picked = active_cards[int(rng.integers(0, len(active_cards)))]
        else:
            picked = active_cards[0]
            best_score = -1e18
            time_decay = 1.0 - 0.45 * (ti * inv_n)

            for cand_c in active_cards:
                gate_c = int(card_gates[cand_c])
                will_elig = (cur_spend[cand_c] + amt) >= gate_c
                spec_c = specs[cand_c]
                b_col = int(spec_c["tx_bucket"][ti])

                imm_gain = 0
                if b_col >= 0:
                    if will_elig:
                        raw_v = int(raw_elig[ti, cand_c])
                    elif cand_c in target_set:
                        raw_v = int(raw_inelig[ti, cand_c])
                    else:
                        raw_v = int(raw_inelig[ti, cand_c])
                    rem_v = int(rem_caps[cand_c][b_col])
                    imm_gain = raw_v if raw_v <= rem_v else rem_v

                score = float(imm_gain)
                if cand_c in target_set and cur_spend[cand_c] < gate_c:
                    needed = min(amt, gate_c - cur_spend[cand_c])
                    score += urgency_weight * float(needed) * time_decay
                    if cur_spend[cand_c] + amt >= gate_c:
                        score += 150.0  # bonus for crossing the threshold right now

                score += float(jitter[ti, cand_c])
                if score > best_score:
                    best_score = score
                    picked = cand_c

        chrom[ti] = picked
        cur_spend[picked] += amt
        spec_p = specs[picked]
        b_p = int(spec_p["tx_bucket"][ti])
        if b_p >= 0:
            elig_now = cur_spend[picked] >= int(card_gates[picked])
            rv = (
                int(raw_elig[ti, picked])
                if elig_now
                else int(raw_inelig[ti, picked])
            )
            rem_v = int(rem_caps[picked][b_p])
            used = rv if rv <= rem_v else rem_v
            if rem_v < INF_LIMIT and used > 0:
                rem_caps[picked][b_p] -= used

    return chrom


def _enumerate_feasible_gate_subsets(
    feasible_gates: list[int],
    card_gates: np.ndarray,
    total_spend: int,
) -> list[list[int]]:
    """Enumerate feasible gate-card subsets whose combined min_spend <= total_spend."""
    if len(feasible_gates) <= 6:
        subsets: list[list[int]] = []
        for r_len in range(len(feasible_gates) + 1):
            for comb in itertools.combinations(feasible_gates, r_len):
                if sum(int(card_gates[c]) for c in comb) <= total_spend:
                    subsets.append(list(comb))
        return subsets

    subsets = [[]]
    for c in feasible_gates:
        subsets.append([c])
    for idx_a in range(len(feasible_gates)):
        for idx_b in range(idx_a + 1, len(feasible_gates)):
            ca = feasible_gates[idx_a]
            cb = feasible_gates[idx_b]
            if int(card_gates[ca] + card_gates[cb]) <= total_spend:
                subsets.append([ca, cb])
    running = 0
    greedy_pack = []
    for c in sorted(feasible_gates, key=lambda x: int(card_gates[x])):
        if running + int(card_gates[c]) <= total_spend:
            greedy_pack.append(c)
            running += int(card_gates[c])
    if len(greedy_pack) >= 3:
        subsets.append(greedy_pack)
    return subsets


def _optimize_month_ga(
    month_txs: list[dict],
    cards: list[str],
    table: dict,
    rng: np.random.Generator,
) -> dict[str, str]:
    """Run the Memetic Genetic Algorithm on a single month's transactions."""
    num_txs = len(month_txs)
    num_cards = len(cards)
    if num_txs == 0:
        return {}
    if num_cards == 1:
        only_card = cards[0]
        return {tx["거래ID"]: only_card for tx in month_txs}

    model = _build_month_model(month_txs, cards, table)
    eval_pop = model["evaluate_population"]
    amounts = model["amounts_i64"]
    total_spend = model["total_spend"]
    raw_elig = model["raw_elig"]
    card_gates = model["card_gates"]
    card_has_unlimited = model["card_has_unlimited"]

    # Fast check: if no card can ever yield > 0 on any transaction even if eligible
    if int(raw_elig.max()) == 0:
        default_card = cards[0]
        return {tx["거래ID"]: default_card for tx in month_txs}

    # Accuracy-first scale parameters
    if num_txs <= 50:
        pop_size, generations, patience = 110, 180, 42
    elif num_txs <= 300:
        pop_size, generations, patience = 132, 210, 46
    elif num_txs < 2000:
        pop_size, generations, patience = 110, 175, 40
    else:
        pop_size, generations, patience = 88, 145, 35

    elite_size = max(4, pop_size // 12)
    feasible_gates = [
        c for c in range(num_cards) if 0 < int(card_gates[c]) <= total_spend
    ]

    # Mutation guidance distribution per transaction
    tx_weights = raw_elig.astype(np.float64) + 25.0
    for c_idx in range(num_cards):
        if card_has_unlimited[c_idx]:
            tx_weights[:, c_idx] += 35.0
    tx_probs = tx_weights / tx_weights.sum(axis=1, keepdims=True)
    tx_cumprobs = np.cumsum(tx_probs, axis=1)

    # -----------------------------------------------------------------
    # Step 1: Diverse Initial Population
    # -----------------------------------------------------------------
    pop = np.empty((pop_size, num_txs), dtype=np.int16)
    n_rand = pop_size // 7
    if num_txs >= 2000:
        n_seeded = min(18, pop_size - n_rand - 8)
        n_weighted = pop_size - n_rand - n_seeded
    else:
        n_weighted = pop_size // 6
        n_seeded = pop_size - n_rand - n_weighted

    pop[:n_rand] = rng.integers(0, num_cards, size=(n_rand, num_txs), dtype=np.int16)

    u_init = rng.random((n_weighted, num_txs))
    pop[n_rand : n_rand + n_weighted] = (
        (u_init[:, :, None] > tx_cumprobs[None, :, :])
        .sum(axis=2)
        .clip(0, num_cards - 1)
        .astype(np.int16)
    )

    key_subsets = _enumerate_feasible_gate_subsets(
        feasible_gates, card_gates, total_spend
    )
    urgency_grid = [0.08, 0.16, 0.05, 0.25]

    for s_idx in range(n_seeded):
        row_idx = n_rand + n_weighted + s_idx
        if s_idx == 0:
            # Near-greedy chronological baseline seed with slight stochastic tie-break
            pop[row_idx] = _construct_subset_seed(
                model,
                list(feasible_gates),
                float(rng.uniform(0.0, 0.01)),
                0.0,
                False,
                rng,
            )
            continue

        adj_idx = s_idx - 1
        if adj_idx < len(key_subsets) * 3:
            chosen_gates = list(key_subsets[adj_idx % len(key_subsets)])
            pass_num = adj_idx // max(1, len(key_subsets))
            base_urg = urgency_grid[pass_num % len(urgency_grid)]
            urg = float(base_urg * rng.uniform(0.85, 1.18))
            eps = float(rng.uniform(0.0, 0.02))
            preassign = (pass_num % 2 == 0)
        else:
            chosen_gates = []
            rem_budget = total_spend
            if feasible_gates:
                perm_fg = rng.permutation(feasible_gates).tolist()
                inc_prob = float(rng.uniform(0.25, 0.85))
                for fg in perm_fg:
                    if int(card_gates[fg]) <= rem_budget and rng.random() < inc_prob:
                        chosen_gates.append(fg)
                        rem_budget -= int(card_gates[fg])
            urg = float(rng.uniform(0.03, 0.32))
            eps = float(rng.uniform(0.002, 0.07))
            preassign = bool(rng.random() < 0.65)

        pop[row_idx] = _construct_subset_seed(
            model, chosen_gates, urg, eps, preassign, rng
        )

    fitness = eval_pop(pop)

    # Polish top 2 initial seeds lightly to jumpstart evolution
    top_init = np.argsort(fitness)[-2:]
    for idx_i in top_init:
        polished_c, polished_f = _memetic_polish(
            pop[int(idx_i)], model, rng, max_rounds=2, enable_swaps=False
        )
        pop[int(idx_i)] = polished_c
        fitness[int(idx_i)] = polished_f

    best_idx = int(np.argmax(fitness))
    best_fit = int(fitness[best_idx])
    best_chrom = pop[best_idx].copy()
    no_improve = 0

    base_mut_rate = float(np.clip(2.0 / num_txs, 0.0002, 0.05))
    num_offspring = pop_size - elite_size
    split_a = num_offspring // 3
    split_b = (2 * num_offspring) // 3

    # -----------------------------------------------------------------
    # Step 2: Generational Evolution Loop
    # -----------------------------------------------------------------
    for gen_idx in range(generations):
        elite_indices = np.argpartition(fitness, -elite_size)[-elite_size:]
        elites = pop[elite_indices].copy()

        # Tournament selection (k = 3)
        cand1 = rng.integers(0, pop_size, size=(num_offspring, 3))
        p1_idx = cand1[np.arange(num_offspring), np.argmax(fitness[cand1], axis=1)]
        cand2 = rng.integers(0, pop_size, size=(num_offspring, 3))
        p2_idx = cand2[np.arange(num_offspring), np.argmax(fitness[cand2], axis=1)]

        parents1 = pop[p1_idx]
        parents2 = pop[p2_idx]
        children = parents1.copy()

        do_cx = rng.random(num_offspring) < 0.88

        # Crossover 1: Prefix/Suffix chronological single-point split
        if split_a > 0 and num_txs >= 2:
            sp_pts = rng.integers(1, num_txs, size=(split_a, 1))
            col_idx = np.arange(num_txs)[None, :]
            sp_mask = (col_idx >= sp_pts) & do_cx[:split_a, None]
            children[:split_a] = np.where(
                sp_mask, parents2[:split_a], parents1[:split_a]
            )

        # Crossover 2: Two-point chronological block crossover
        mid_len = split_b - split_a
        if mid_len > 0 and num_txs >= 2:
            tp_pts = np.sort(rng.integers(0, num_txs + 1, size=(mid_len, 2)), axis=1)
            col_idx = np.arange(num_txs)[None, :]
            tp_mask = (
                (col_idx >= tp_pts[:, 0:1])
                & (col_idx < tp_pts[:, 1:2])
                & do_cx[split_a:split_b, None]
            )
            children[split_a:split_b] = np.where(
                tp_mask, parents2[split_a:split_b], parents1[split_a:split_b]
            )

        # Crossover 3: Uniform crossover
        tail_len = num_offspring - split_b
        if tail_len > 0:
            u_mask = (rng.random((tail_len, num_txs)) < 0.5) & do_cx[split_b:, None]
            children[split_b:] = np.where(
                u_mask, parents2[split_b:], parents1[split_b:]
            )

        # Adaptive sparse point mutation
        mut_scale = 1.0 + min(1.8, no_improve / max(8, patience // 2))
        mut_rate = min(0.08, base_mut_rate * mut_scale)

        mut_rows, mut_cols = np.nonzero(
            rng.random((num_offspring, num_txs)) < mut_rate
        )
        num_muts = len(mut_rows)
        if num_muts > 0:
            use_rand = rng.random(num_muts) < 0.40
            rand_picks = rng.integers(0, num_cards, size=num_muts, dtype=np.int16)
            u_vals = rng.random(num_muts)[:, None]
            guided_picks = (
                (u_vals > tx_cumprobs[mut_cols])
                .sum(axis=1)
                .clip(0, num_cards - 1)
                .astype(np.int16)
            )
            children[mut_rows, mut_cols] = np.where(use_rand, rand_picks, guided_picks)

        # Macro mutation: Gate-Boost / Evacuate & Early Zero-Benefit Shift
        macro_prob = 0.05 if num_txs >= 2000 else 0.22
        if feasible_gates and total_spend < 5 * int(card_gates.sum()):
            macro_rows = np.flatnonzero(rng.random(num_offspring) < macro_prob)
            for r_idx in macro_rows:
                ch = children[r_idx]
                sp = np.bincount(ch, weights=amounts, minlength=num_cards).astype(
                    np.int64
                )
                under = [
                    c for c in feasible_gates if 0 < sp[c] < int(card_gates[c])
                ]
                if under:
                    target_c = under[int(rng.integers(0, len(under)))]
                    deficit = int(card_gates[target_c] - sp[target_c])
                    if rng.random() < 0.62:
                        # Boost target_c preferentially using earlier / low-benefit txs
                        cand_tx = np.flatnonzero(ch != target_c)
                        if len(cand_tx) > 0:
                            # Score candidates by (raw_elig_on_cur / amt) + chronological bias
                            cur_cards = ch[cand_tx]
                            opp_ratio = raw_elig[cand_tx, cur_cards] / (
                                amounts[cand_tx] + 1.0
                            )
                            time_bias = np.arange(len(cand_tx), dtype=np.float64) / max(
                                1, len(cand_tx)
                            )
                            order_key = (
                                opp_ratio
                                + 0.04 * time_bias
                                + rng.uniform(0.0, 0.02, size=len(cand_tx))
                            )
                            for idx_ord in np.argsort(order_key):
                                ti = int(cand_tx[idx_ord])
                                src = int(ch[ti])
                                amt_i = int(amounts[ti])
                                if (
                                    card_gates[src] > 0
                                    and sp[src] >= card_gates[src]
                                    and sp[src] - amt_i < card_gates[src]
                                ):
                                    continue
                                ch[ti] = target_c
                                sp[src] -= amt_i
                                sp[target_c] += amt_i
                                deficit -= amt_i
                                if deficit <= 0:
                                    break
                    else:
                        # Evacuate dead gate card to active/zero-gate cards
                        stuck_tx = np.flatnonzero(ch == target_c)
                        alt_cards = [
                            c
                            for c in range(num_cards)
                            if c != target_c
                            and (card_gates[c] == 0 or sp[c] >= card_gates[c])
                        ]
                        if not alt_cards:
                            alt_cards = [c for c in range(num_cards) if c != target_c]
                        if len(stuck_tx) > 0 and alt_cards:
                            # Pick highest raw_elig among alt_cards with slight randomness
                            alt_arr = np.array(alt_cards, dtype=np.int16)
                            sub_raw = raw_elig[stuck_tx][:, alt_arr]
                            best_local = np.argmax(
                                sub_raw
                                + rng.uniform(0.0, 5.0, size=sub_raw.shape),
                                axis=1,
                            )
                            ch[stuck_tx] = alt_arr[best_local]

        pop[:elite_size] = elites
        pop[elite_size:] = children
        fitness = eval_pop(pop)

        # Periodic Lamarckian memetic polish on top elite
        polish_interval = 50 if num_txs >= 2000 else 35
        if (gen_idx + 1) % polish_interval == 0:
            cur_top = int(np.argmax(fitness))
            pol_c, pol_f = _memetic_polish(
                pop[cur_top],
                model,
                rng,
                max_rounds=(1 if num_txs >= 2000 else 3),
                enable_swaps=False,
            )
            pop[cur_top] = pol_c
            fitness[cur_top] = pol_f

        gen_best_idx = int(np.argmax(fitness))
        gen_best_fit = int(fitness[gen_best_idx])
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_chrom = pop[gen_best_idx].copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # -----------------------------------------------------------------
    # Step 3: Final Memetic Refinement on Top Distinct Elites
    # -----------------------------------------------------------------
    max_elites_to_polish = 1 if num_txs >= 2000 else 3
    polish_rounds = 2 if num_txs >= 2000 else 6
    sorted_indices = np.argsort(fitness)[::-1]
    distinct_candidates: list[np.ndarray] = [best_chrom.copy()]
    for idx_s in sorted_indices:
        cand_chrom = pop[int(idx_s)]
        if not any(np.array_equal(cand_chrom, prev) for prev in distinct_candidates):
            distinct_candidates.append(cand_chrom.copy())
        if len(distinct_candidates) >= max_elites_to_polish:
            break

    for cand_chrom in distinct_candidates:
        refined_chrom, refined_fit = _memetic_polish(
            cand_chrom,
            model,
            rng,
            max_rounds=polish_rounds,
            enable_swaps=(num_txs <= 1500),
        )
        if refined_fit > best_fit:
            best_fit = refined_fit
            best_chrom = refined_chrom

    # If the entire month cannot unlock any positive benefit (best_fit == 0),
    # assign each transaction to the card with highest potential raw benefit.
    if best_fit == 0:
        best_chrom = np.argmax(raw_elig, axis=1).astype(np.int16)

    return {
        month_txs[ti]["거래ID"]: cards[int(best_chrom[ti])] for ti in range(num_txs)
    }


# =====================================================================
# 4. Public Interface Entry Point (INTERFACE.md Compliant)
# =====================================================================


def assign(
    transactions: list[dict],
    cards: list[str],
    benefit_db: dict,
    *,
    seed: int | None = None,
) -> dict:
    """
    Assign transactions to held cards to maximize total benefit using a
    Memetic Genetic Algorithm.

    Args:
        transactions: list of {"거래ID": str, "결제일시": str, "업종": str, "결제금액": int}
        cards: list of held card IDs (e.g., ["CARD01", "CARD04", "CARD08"])
        benefit_db: dictionary loaded from data/cards/card_benefits.json
        seed: optional RNG seed (None preserves natural GA stochasticity;
              an integer ensures deterministic reproducibility)

    Returns:
        {
            "배정결과": [{"거래ID": str, "카드ID": str}, ...],
            "총혜택액": int,
        }
    """
    if not cards:
        raise ValueError("cards must contain at least one card ID.")
    if not transactions:
        return {"배정결과": [], "총혜택액": 0}

    table = build_benefit_table(cards, benefit_db)
    cleaned = clean_transactions(transactions)
    if not cleaned:
        return format_result(transactions, {}, cards[0], 0)

    rng = np.random.default_rng(seed)

    full_assignment: dict[str, str] = {}
    for month_txs in split_by_month(cleaned).values():
        month_assignment = _optimize_month_ga(month_txs, cards, table, rng)
        full_assignment.update(month_assignment)

    _, total = evaluate_assignment(cleaned, full_assignment, table)
    return format_result(transactions, full_assignment, cards[0], total)


if __name__ == "__main__":
    import csv
    import json
    import statistics
    import time
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]

    with open(root / "data/cards/card_benefits.json", encoding="utf-8") as db_file:
        loaded_benefit_db = json.load(db_file)
    with open(
        root / "experiments/카드_조합_실험계획.json", encoding="utf-8"
    ) as plan_file:
        combo_plan = json.load(plan_file)

    def read_csv_rows(csv_path: Path) -> list[dict]:
        with open(csv_path, encoding="utf-8-sig") as csv_file:
            return list(csv.DictReader(csv_file))

    datasets = {
        "실제 93건": read_csv_rows(root / "data/real/실제결제내역_전처리.csv")
    }
    for syn_path in sorted((root / "data/synthetic").glob("*.csv")):
        datasets[syn_path.stem] = read_csv_rows(syn_path)

    for dataset_name, dataset_txs in datasets.items():
        print(f"\n===== {dataset_name} =====")
        for k_key in ["2", "3", "4", "8"]:
            totals: list[int] = []
            start_time = time.perf_counter()
            for card_subset in combo_plan[k_key]:
                res = assign(dataset_txs, card_subset, loaded_benefit_db)
                totals.append(res["총혜택액"])
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            mean_total = sum(totals) / len(totals)
            print(
                f"카드 {k_key}장 조합 {len(totals):2d}개 | 평균 총혜택액 {mean_total:>9,.0f}원 | "
                f"최소 {min(totals):>7,}원 | 최대 {max(totals):>7,}원 | {elapsed_ms:7.1f}ms"
            )

    repeated_totals = [
        assign(datasets["실제 93건"], combo_plan["3"][0], loaded_benefit_db)["총혜택액"]
        for _ in range(10)
    ]
    print(
        f"\n[10-Run Repeatability] 실제 93건 · cards {combo_plan['3'][0]} -> "
        f"mean {statistics.mean(repeated_totals):,.1f}원, "
        f"stdev {statistics.stdev(repeated_totals):,.1f}원 "
        f"(min {min(repeated_totals):,}원 ~ max {max(repeated_totals):,}원)"
    )

    sample_result = assign(
        datasets["실제 93건"], combo_plan["3"][0], loaded_benefit_db, seed=42
    )
    print(
        f"\n[Example (seed=42)] cards {combo_plan['3'][0]} -> 총혜택액 {sample_result['총혜택액']:,}원"
    )
    for out_row in sample_result["배정결과"][:5]:
        print("  ", out_row)
