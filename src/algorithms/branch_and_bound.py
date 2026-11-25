import time
from bisect import bisect_left

from common import clean_transactions, build_benefit_table, evaluate_assignment, split_by_month, format_result

# 결제를 날짜순으로 한 건씩 카드에 배정하는 DFS. 지금까지 혜택 + 상한 <= 현재 최적해면 가지치기
# 혜택은 common.evaluate_assignment로 결제x카드별 값을 미리 구해두고, 탐색 중엔 남은 한도만 반영
MAX_NODES = 300_000  # 월별 탐색 노드 상한 (dp.py의 MAX_STATES와 같은 안전장치)
MAX_SECONDS = 30.0   # 월별 탐색 시간 상한
TIME_CHECK_EVERY = 256
FLOAT_SLACK = 0.01   # 실수로 계산한 상한을 내림할 때 오차 여유 (혜택은 원 단위 정수)
ROOT_ROUNDS = 10     # 루트에서 라그랑주 승수 갱신 횟수 (자식은 부모 값을 이어받아 1회)
NO_LIMIT = float("inf")


def _precompute_gains(transactions: list[dict], cards: list[str], table: dict):
    """결제x카드별 (전월실적 충족 시 혜택, 미충족 시 혜택, 한도키). 한도는 무시하고 계산."""
    unlimited = {key: None for key in table["limits"]}
    eligible_table = {"benefits": table["benefits"], "limits": unlimited, "gates": {}}
    ineligible_table = {
        "benefits": table["benefits"],
        "limits": unlimited,
        "gates": {card: float("inf") for card in cards if card in table["gates"]},
    }

    gain_eligible, gain_ineligible, gain_key = [], [], []
    for tx in transactions:
        row_e, row_n, row_k = [], [], []
        for card in cards:
            single = {tx["거래ID"]: card}
            _, g_e = evaluate_assignment([tx], single, eligible_table)
            _, g_n = evaluate_assignment([tx], single, ineligible_table)
            keys = {c["limit_key"] for c in table["benefits"][card].get(tx["업종"], [])}
            assert len(keys) <= 1, "같은 (카드, 업종) 혜택이 한도키를 공유한다는 common.py 전제가 깨짐"
            row_e.append(g_e)
            row_n.append(g_n)
            row_k.append(next(iter(keys)) if keys else None)
        gain_eligible.append(row_e)
        gain_ineligible.append(row_n)
        gain_key.append(row_k)
    return gain_eligible, gain_ineligible, gain_key


def _dominates(a, b) -> bool:
    """a가 혜택·한도잔액·전월실적 누적 모두 b 이상이면 b는 볼 필요 없음."""
    total_a, limits_a, spend_a = a
    total_b, limits_b, spend_b = b
    return (total_a >= total_b
            and all(x >= y for x, y in zip(limits_a, limits_b))
            and all(x >= y for x, y in zip(spend_a, spend_b)))


class _MonthSearch:
    """한 달치 결제에 대한 분기한정 탐색."""

    def __init__(self, transactions: list[dict], cards: list[str], table: dict):
        self.transactions = transactions
        self.cards = cards
        self.n = n = len(transactions)
        self.num_cards = m = len(cards)

        limit_keys = sorted(table["limits"])
        key_index = {key: k for k, key in enumerate(limit_keys)}
        self.init_limits = tuple(
            NO_LIMIT if table["limits"][key] is None else table["limits"][key] for key in limit_keys)
        self.is_limited = [table["limits"][key] is not None for key in limit_keys]
        self.limited_keys = [k for k in range(len(limit_keys)) if self.is_limited[k]]

        # 전월실적 조건이 있는 카드만 누적액을 상태로 관리
        self.gated_cards = [c for c in range(m) if cards[c] in table["gates"]]
        self.gate_index = {c: g for g, c in enumerate(self.gated_cards)}
        self.gate_values = [table["gates"][cards[c]] for c in self.gated_cards]

        self.gain_e, self.gain_n, gain_key = _precompute_gains(transactions, cards, table)
        # 결제 t를 카드 c로 냈을 때 차감되는 한도키 (-1: 혜택 없음)
        self.key_of = [[key_index[key] if key is not None else -1 for key in row] for row in gain_key]
        self.key_card = [cards.index(key.split(":", 1)[0]) for key in limit_keys]
        self.key_txs = [[] for _ in limit_keys]
        for t in range(n):
            for c in range(m):
                k = self.key_of[t][c]
                if k >= 0 and self.gain_e[t][c] > 0:
                    self.key_txs[k].append(t)

        # 결제마다 최대 혜택(한도·전월실적 무시)의 접미사 합 = 가장 단순한 상한
        self.suffix_best = [0] * (n + 1)
        for t in range(n - 1, -1, -1):
            self.suffix_best[t] = self.suffix_best[t + 1] + max(self.gain_e[t])
        self.prefix_amount = [0] * (n + 1)
        for t in range(n):
            self.prefix_amount[t + 1] = self.prefix_amount[t] + transactions[t]["금액"]

    def _eligible_from(self, i, spend):
        """카드별로 전월실적을 가장 빨리 채울 수 있는 결제 인덱스 (n이면 이번 달엔 불가)."""
        taus = [i] * self.num_cards
        for g, c in enumerate(self.gated_cards):
            if spend[g] < self.gate_values[g]:
                need = self.prefix_amount[i] + self.gate_values[g] - spend[g]
                taus[c] = min(bisect_left(self.prefix_amount, need) - 1, self.n)
        return taus

    def _lagrangian_bound(self, i, limits, spend, lam, rounds):
        """라그랑주 완화 상한: 월한도 제약을 승수 λ로 완화
        UB(λ) = Σ λ_k·남은한도_k + Σ_t max_c 혜택_tc·(1-λ_k)"""
        n, m = self.n, self.num_cards
        taus = self._eligible_from(i, spend)
        gain_e, gain_n, key_of = self.gain_e, self.gain_n, self.key_of
        eff = [None] * n
        for t in range(i, n):
            eff[t] = [gain_e[t][c] if t >= taus[c] else gain_n[t][c] for c in range(m)]
        lam = list(lam)

        for _ in range(rounds):
            for k in self.limited_keys:
                c_k = self.key_card[k]
                items = []
                for t in self.key_txs[k]:
                    if t < i or eff[t][c_k] <= 0:
                        continue
                    g = eff[t][c_k]
                    other = 0.0
                    for d in range(m):
                        if d != c_k:
                            kd = key_of[t][d]
                            v = eff[t][d] * (1.0 - lam[kd] if kd >= 0 else 1.0)
                            if v > other:
                                other = v
                    items.append((1.0 - other / g, g))
                items.sort(reverse=True)
                capacity, new_lam = limits[k], 0.0
                for breakpoint, g in items:
                    capacity -= g
                    if capacity < 0:
                        new_lam = max(0.0, breakpoint)
                        break
                lam[k] = new_lam

        # 아직 전월실적 미달이지만 이번 달 안에 채울 수 있는 카드
        pending = [c for c in self.gated_cards
                   if spend[self.gate_index[c]] < self.gate_values[self.gate_index[c]] and taus[c] < n]
        value = sum(lam[k] * limits[k] for k in self.limited_keys)
        if not pending:
            for t in range(i, n):
                row, keys, best = eff[t], key_of[t], 0.0
                for c in range(m):
                    k = keys[c]
                    v = row[c] * (1.0 - lam[k] if k >= 0 else 1.0)
                    if v > best:
                        best = v
                value += best
            return int(value + FLOAT_SLACK), tuple(lam)
        return self._bound_with_gate_cost(i, spend, lam, eff, taus, pending, value), tuple(lam)

    def _bound_with_gate_cost(self, i, spend, lam, eff, taus, pending, value):
        """전월실적 미달 카드는 (a) 끝까지 미달해 실적조건 혜택을 잃거나
        (b) 다른 카드 혜택을 포기하고 결제를 몰아 실적을 채워야 함.
        어떤 카드가 실적을 채우는지(시나리오)별로 손실·비용을 빼고 그중 최댓값을 상한으로 씀."""
        n, m, key_of, gain_n = self.n, self.num_cards, self.key_of, self.gain_n
        txs = self.transactions
        pending_set = set(pending)
        pending_pos = {c: j for j, c in enumerate(pending)}
        loss_at = {c: [0.0] * (n - i) for c in pending}      # 실적조건 혜택을 잃을 때 감소량
        cost_at = {c: [0.0] * (n - i) for c in pending}      # 결제를 c로 몰 때 비용 (실적 충족 후)
        cost_before = {c: [0.0] * (n - i) for c in pending}  # 결제를 c로 몰 때 비용 (실적 충족 전)
        num_masks = 1 << len(pending)
        value_failed = [value] * num_masks  # mask의 카드들이 실적 미달일 때 상한
        for t in range(i, n):
            row, keys = eff[t], key_of[t]
            vals, degraded = [0.0] * m, [0.0] * m
            for c in range(m):
                factor = 1.0 - lam[keys[c]] if keys[c] >= 0 else 1.0
                vals[c] = row[c] * factor
                degraded[c] = gain_n[t][c] * factor if c in pending_set and t >= taus[c] else vals[c]
            best = max(0.0, max(vals))
            lowest = max(0.0, max(degraded))
            for mask in range(num_masks):
                v = 0.0
                for c in range(m):
                    j = pending_pos.get(c)
                    x = degraded[c] if j is not None and mask >> j & 1 else vals[c]
                    if x > v:
                        v = x
                value_failed[mask] += v
            for c in pending:
                if degraded[c] != vals[c]:
                    without_c = max(0.0, max(degraded[c] if d == c else vals[d] for d in range(m)))
                    loss_at[c][t - i] = best - without_c
                cost_at[c][t - i] = max(0.0, lowest - vals[c])
                cost_before[c][t - i] = max(0.0, lowest - degraded[c])

        def cover_cost(costs, end, need, costs_late=None, late_from=None):
            """결제 i..end-1로 금액 need를 채우는 최소 비용 (분할 배낭 완화)."""
            if costs_late is not None:
                costs = [costs[j] if i + j < late_from else costs_late[j] for j in range(end - i)]
            items = sorted((costs[j] / txs[i + j]["금액"], j) for j in range(end - i))
            total = 0.0
            for _, j in items:
                amount = txs[i + j]["금액"]
                if amount >= need:
                    return total + costs[j] * need / amount
                total += costs[j]
                need -= amount
            return float("inf")

        def timed_cost(c, need):
            """실적을 넘는 시점 T에 대해 min(T 이전 손실 + T까지 결제로 채우는 비용). T는 구간 단위로 계산."""
            start = taus[c]
            if start >= n:
                return float("inf")
            span = n - start
            step = 1 if span <= 48 else -(-span // 24)
            prefix_loss = [0.0] * (n - i + 1)
            for j in range(n - i):
                prefix_loss[j + 1] = prefix_loss[j] + loss_at[c][j]
            best = float("inf")
            for a in range(start, n, step):
                b = min(a + step - 1, n - 1)
                best = min(best, prefix_loss[a - i] + cover_cost(cost_before[c], b + 1, need, cost_at[c], a))
            return best

        needs = [self.gate_values[self.gate_index[c]] - spend[self.gate_index[c]] for c in pending]
        costs = [timed_cost(c, need) for c, need in zip(pending, needs)]

        remaining_amount = self.prefix_amount[n] - self.prefix_amount[i]
        best_scenario = None
        for mask in range(1 << len(pending)):
            chosen = [j for j in range(len(pending)) if mask >> j & 1]
            need_sum = sum(needs[j] for j in chosen)
            if need_sum > remaining_amount:
                continue  # 남은 결제로 이 카드들을 동시에 채울 수 없음
            cost = sum(costs[j] for j in chosen)
            if len(chosen) >= 2:
                # 여러 장을 채울 땐 같은 결제를 중복으로 못 씀
                joint = [min(cost_at[pending[j]][idx] for j in chosen) for idx in range(n - i)]
                cost = max(cost, cover_cost(joint, n, need_sum))
            scenario = value_failed[(num_masks - 1) ^ mask] - cost
            if best_scenario is None or scenario > best_scenario:
                best_scenario = scenario
        return int(best_scenario + FLOAT_SLACK)

    def _children(self, i, total, limits, spend):
        amount = self.transactions[i]["금액"]
        children = []
        for c in range(self.num_cards):
            g = self.gate_index.get(c)
            new_spend = spend
            if g is None:
                eligible = True
            else:
                eligible = spend[g] + amount >= self.gate_values[g]
                clamped = min(spend[g] + amount, self.gate_values[g])  # 기준 넘으면 더 쌓을 필요 없음
                if clamped != spend[g]:
                    new_spend = spend[:g] + (clamped,) + spend[g + 1:]

            gain = self.gain_e[i][c] if eligible else self.gain_n[i][c]
            new_limits = limits
            k = self.key_of[i][c]
            if k >= 0 and gain > 0:
                gain = min(gain, limits[k])
                if self.is_limited[k] and gain > 0:
                    new_limits = limits[:k] + (limits[k] - gain,) + limits[k + 1:]
            children.append((total + gain, new_limits, new_spend, c))

        # 다른 자식보다 모든 면에서 못한 자식은 제거
        kept = []
        for child in children:
            if any(_dominates(other[:3], child[:3]) for other in kept):
                continue
            kept = [o for o in kept if not _dominates(child[:3], o[:3])]
            kept.append(child)
        # 혜택 큰 자식부터 탐색하도록 작은 것부터 push
        kept.sort(key=lambda ch: (ch[0], sum(ch[2])))
        return kept

    def run(self) -> tuple[dict, int]:
        n = self.n
        if n == 0:
            return {}, 0
        init_spend = tuple(0 for _ in self.gated_cards)
        init_lam = tuple(0.0 for _ in self.init_limits)

        best_total, best_path = -1, None
        best_seen = {}  # 같은 상태 재탐색 방지
        path = [0] * n
        nodes = 0
        started = time.perf_counter()
        stack = [(0, 0, self.init_limits, init_spend, None, init_lam)]
        while stack:
            i, total, limits, spend, chosen, lam = stack.pop()
            if chosen is not None:
                path[i - 1] = chosen
            nodes += 1
            if nodes > MAX_NODES:
                raise ValueError(
                    f"분기한정법 노드 수 초과: 결제 {n}건 x 카드 {self.num_cards}장, 탐색 노드 {MAX_NODES:,}개 초과 "
                    f"— 카드 조합을 줄이거나(2~3장) 결제 건수를 줄여서 다시 시도할 것"
                )
            if nodes % TIME_CHECK_EVERY == 0 and time.perf_counter() - started > MAX_SECONDS:
                raise ValueError(
                    f"분기한정법 시간 초과: 결제 {n}건 x 카드 {self.num_cards}장, 월별 {MAX_SECONDS:.0f}초 초과 "
                    f"(탐색 노드 {nodes:,}개) — 카드 조합을 줄이거나(2~3장) 결제 건수를 줄여서 다시 시도할 것"
                )

            if i == n:
                if total > best_total:
                    best_total, best_path = total, path[:]
                continue

            # 가지치기: 단순 상한 → 같은 상태 → 라그랑주 상한 순
            if total + self.suffix_best[i] <= best_total:
                continue
            state = (i, limits, spend)
            if best_seen.get(state, -1) >= total:
                continue
            best_seen[state] = total
            if best_total >= 0:
                bound, lam = self._lagrangian_bound(i, limits, spend, lam, ROOT_ROUNDS if i == 0 else 1)
                if total + bound <= best_total:
                    continue

            for new_total, new_limits, new_spend, c in self._children(i, total, limits, spend):
                stack.append((i + 1, new_total, new_limits, new_spend, c, lam))

        assignment = {tx["거래ID"]: self.cards[c] for tx, c in zip(self.transactions, best_path)}
        return assignment, best_total


def _search_month(transactions: list[dict], cards: list[str], table: dict) -> tuple[dict, int]:
    return _MonthSearch(transactions, cards, table).run()


def assign(transactions: list[dict], cards: list[str], benefit_db: dict) -> dict:
    if not cards:
        raise ValueError("카드가 최소 1장은 있어야 함")
    table = build_benefit_table(cards, benefit_db)
    cleaned = clean_transactions(transactions)
    if len({tx["거래ID"] for tx in cleaned}) != len(cleaned):
        raise ValueError("거래ID가 중복됨")

    full_assignment = {}
    for month_txs in split_by_month(cleaned).values():
        assignment, searched_total = _search_month(month_txs, cards, table)
        _, checked_total = evaluate_assignment(month_txs, assignment, table)
        assert searched_total == checked_total, "탐색 중 혜택 계산이 common.evaluate_assignment와 불일치"
        full_assignment.update(assignment)

    _, total = evaluate_assignment(cleaned, full_assignment, table)
    return format_result(transactions, full_assignment, cards[0], total)


def _verify_month(transactions: list[dict], cards: list[str], table: dict) -> int:
    """소규모 검증용: 완전탐색 결과와 비교."""
    from itertools import product

    assignment, total = _search_month(transactions, cards, table)
    ids = [tx["거래ID"] for tx in transactions]
    expected = max(evaluate_assignment(transactions, dict(zip(ids, choices)), table)[1]
                   for choices in product(cards, repeat=len(ids)))
    actual = evaluate_assignment(transactions, assignment, table)[1]
    if total != expected or actual != expected:
        raise AssertionError(f"분기한정법={total}, 재채점={actual}, 완전탐색={expected}")
    return expected


if __name__ == "__main__":
    import argparse
    import csv
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=int, default=0, metavar="N",
                        help="데이터셋별 앞 N건(1~8)으로 완전탐색과 최적해 일치 여부만 확인하고 종료")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]

    with open(root / "data/cards/card_benefits.json", encoding="utf-8") as f:
        benefit_db = json.load(f)
    with open(root / "experiments/카드_조합_실험계획.json", encoding="utf-8") as f:
        combo_plan = json.load(f)

    def read_csv(path):
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    datasets = {"실제 93건": read_csv(root / "data/real/실제결제내역_전처리.csv")}
    for path in sorted((root / "data/synthetic").glob("*.csv")):
        datasets[path.stem] = read_csv(path)

    if args.verify:
        if not 1 <= args.verify <= 8:
            parser.error("--verify N 은 1~8 사이")
        for name, transactions in datasets.items():
            sample = clean_transactions(transactions[:args.verify])
            for k in ["2", "3"]:
                for cards in combo_plan[k]:
                    table = build_benefit_table(cards, benefit_db)
                    for month_txs in split_by_month(sample).values():
                        _verify_month(month_txs, cards, table)
            print(f"{name}: 앞 {args.verify}건, 카드 2~3장 조합 전부 완전탐색과 일치")
        raise SystemExit

    for name, transactions in datasets.items():
        print(f"\n===== {name} =====")
        for k in ["2", "3", "4", "8"]:
            totals, failed, start = [], 0, time.perf_counter()
            for cards in combo_plan[k]:
                try:
                    totals.append(assign(transactions, cards, benefit_db)["총혜택액"])
                except ValueError:
                    failed += 1  # 노드 상한 초과 (의도된 안전장치)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if not totals:
                print(f"카드 {k}장 조합 {failed:2d}개 전부 노드 상한 초과 | {elapsed_ms:9.1f}ms")
                continue
            avg = sum(totals) / len(totals)
            note = f" | 노드 상한 초과 {failed}개 제외" if failed else ""
            print(f"카드 {k}장 조합 {len(totals):2d}개 | 평균 총혜택액 {avg:>9,.0f}원 | "
                  f"최소 {min(totals):>7,}원 | 최대 {max(totals):>7,}원 | {elapsed_ms:9.1f}ms{note}")

    example = assign(datasets["실제 93건"], combo_plan["3"][0], benefit_db)
    print(f"\n[예시] 카드 {combo_plan['3'][0]} -> 총혜택액 {example['총혜택액']:,}원")
    for row in example["배정결과"][:5]:
        print("  ", row)
