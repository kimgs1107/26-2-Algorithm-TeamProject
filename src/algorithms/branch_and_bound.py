import time
from bisect import bisect_left

from common import clean_transactions, build_benefit_table, evaluate_assignment, split_by_month, format_result

# 분기한정법(Branch and Bound): 결제를 날짜순으로 한 건씩 카드에 배정하는 DFS 탐색 트리
#   - 깊이 i 노드 = 앞의 i건까지 배정이 끝난 상태 (한도 잔액, 카드별 전월실적 누적)
#   - 상한(upper bound) = 남은 결제에서 얻을 수 있는 혜택의 낙관적 추정치 (실제 최적값 이상이 보장됨)
#   - 지금까지 혜택 + 상한 <= 현재까지 찾은 최적해(incumbent)면 그 가지는 더 볼 필요 없음 → 가지치기
#
# 혜택 계산은 common.evaluate_assignment를 그대로 사용한다.
#   결제 1건 x 카드 1장마다 evaluate_assignment를 두 번 호출해서
#   "전월실적 충족 시 혜택(한도 무시)" / "미충족 시 혜택(한도 무시)"을 미리 구해 두고,
#   탐색 중에는 거기에 남은 한도만 min으로 씌운다.
#   (common.py에서 같은 (카드, 업종)의 혜택들은 한도키 하나를 공유하므로
#    max_후보 min(혜택_후보, 잔액) == min(max_후보 혜택_후보, 잔액) 이 성립)
#   월별 최종 총혜택액은 evaluate_assignment로 다시 채점하고, 탐색값과 일치하는지 assert로 확인한다.
MAX_NODES = 300_000  # 월별 탐색 노드 상한 (결제 건수·카드 수가 크면 트리가 지수적으로 커지므로 안전장치)
MAX_SECONDS = 30.0   # 월별 탐색 시간 상한 (노드 하나 처리 비용이 커서 노드 상한 전에 오래 걸리는 경우 대비)
TIME_CHECK_EVERY = 256  # 시간 확인 주기(노드 수) — 매 노드마다 시계를 보면 오히려 느려짐
# 상한은 실수로 계산한 뒤 내림하는데, 실제 혜택은 정수(원)이므로 1원 미만 여유를 둬도 상한으로 안전.
# 결제가 많을 때 실수 덧셈 오차가 쌓여도 상한이 실제 최적값보다 작아지지 않도록 넉넉히 0.01원으로 둠.
FLOAT_SLACK = 0.01
ROOT_ROUNDS = 10     # 루트에서 라그랑주 승수를 다듬는 반복 횟수 (자식 노드는 부모 승수를 이어받아 1회만 갱신)
NO_LIMIT = float("inf")


def _precompute_gains(transactions: list[dict], cards: list[str], table: dict):
    """결제 t, 카드 c 별로 (전월실적 충족 시 혜택, 미충족 시 혜택, 한도키)를 evaluate_assignment로 계산."""
    unlimited = {key: None for key in table["limits"]}
    eligible_table = {"benefits": table["benefits"], "limits": unlimited, "gates": {}}
    # 전월실적 기준을 절대 못 넘는 값으로 두면 전월실적 조건 없는 혜택만 남음
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
    """자식 노드 a가 b보다 모든 면에서 같거나 나으면(혜택·한도잔액·전월실적 누적) b는 볼 필요 없음."""
    total_a, limits_a, spend_a = a
    total_b, limits_b, spend_b = b
    return (total_a >= total_b
            and all(x >= y for x, y in zip(limits_a, limits_b))
            and all(x >= y for x, y in zip(spend_a, spend_b)))


class _MonthSearch:
    """한 달치 결제에 대한 분기한정 탐색. 상한 계산에 쓰는 사전 계산 테이블을 들고 있음."""

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
        # 결제 t를 카드 c로 냈을 때 차감되는 한도키 인덱스 (-1: 해당 업종 혜택 없음)
        self.key_of = [[key_index[key] if key is not None else -1 for key in row] for row in gain_key]
        # 한도키 k가 어떤 카드 소속이고, 어떤 결제들이 그 키로 혜택을 받을 수 있는지
        self.key_card = [cards.index(key.split(":", 1)[0]) for key in limit_keys]
        self.key_txs = [[] for _ in limit_keys]
        for t in range(n):
            for c in range(m):
                k = self.key_of[t][c]
                if k >= 0 and self.gain_e[t][c] > 0:
                    self.key_txs[k].append(t)

        # 결제마다 가장 좋은 카드를 한도·전월실적 무시하고 고른 혜택의 접미사 합 (가장 싸고 느슨한 상한)
        self.suffix_best = [0] * (n + 1)
        for t in range(n - 1, -1, -1):
            self.suffix_best[t] = self.suffix_best[t + 1] + max(self.gain_e[t])
        self.prefix_amount = [0] * (n + 1)
        for t in range(n):
            self.prefix_amount[t + 1] = self.prefix_amount[t] + transactions[t]["금액"]

    # ------------------------------------------------------------------ 상한
    def _eligible_from(self, i, spend):
        """카드별로 '남은 결제를 전부 몰아줬을 때 전월실적 기준을 처음 넘을 수 있는 결제 인덱스'.
        그 전 결제는 아무리 잘 배정해도 미충족 혜택만 받을 수 있음 (n이면 이번 달 안에 못 채움)."""
        taus = [i] * self.num_cards
        for g, c in enumerate(self.gated_cards):
            if spend[g] < self.gate_values[g]:
                need = self.prefix_amount[i] + self.gate_values[g] - spend[g]
                taus[c] = min(bisect_left(self.prefix_amount, need) - 1, self.n)
        return taus

    def _lagrangian_bound(self, i, limits, spend, lam, rounds):
        """라그랑주 완화 상한: 월한도 제약을 승수 λ_k(0~1)로 목적함수에 넣고, '결제당 카드 1장' 제약은 유지.
            UB(λ) = Σ_k λ_k·남은한도_k + Σ_t max(0, max_c 혜택_tc·(1-λ_k(t,c)))
        어떤 λ를 써도 실제 최적값 이상이므로(약쌍대성) 상한으로 유효하고, λ를 잘 고를수록 상한이 조여진다.
        λ는 한도키별 좌표하강으로 갱신: 다른 키를 고정하면 λ_k의 최적값은 분할 배낭 문제의 임계 비율과 같음."""
        n, m = self.n, self.num_cards
        taus = self._eligible_from(i, spend)
        gain_e, gain_n, key_of = self.gain_e, self.gain_n, self.key_of
        # eff[t][c]: 이 노드 기준으로 낙관적으로 받을 수 있는 혜택(한도 무시, 전월실적은 가능한 시점부터 충족)
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
                    other = 0.0  # 이 결제를 다른 카드로 냈을 때 (승수 반영) 가장 좋은 값
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

        # 아직 전월실적을 못 채웠지만 이번 달 안에 채울 수는 있는 카드
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
            return int(value + FLOAT_SLACK), tuple(lam)  # 실제 혜택은 정수이므로 내림해도 상한으로 유효
        return self._bound_with_gate_cost(i, spend, lam, eff, taus, pending, value), tuple(lam)

    def _bound_with_gate_cost(self, i, spend, lam, eff, taus, pending, value):
        """전월실적 미충족 카드 c마다 둘 중 하나는 반드시 일어남:
          (a) 이번 달 끝까지 기준 미달 → c의 실적조건 혜택을 전부 잃음 (손실 loss_c)
          (b) 어떤 결제 T에서 기준을 넘음 → T까지의 결제 중 금액 합이 (기준-현재누적) 이상이 되도록 c로 몰아야 하고
              (그 결제들은 원래 최선 카드 대신 c를 씀: 비용), T 이전 결제에서는 c의 실적조건 혜택을 못 받음(손실)
        시나리오(어떤 카드들이 기준을 채우는지)마다 해당 손실·비용을 빼고, 가능한 시나리오 중 최댓값을 상한으로 씀.
          - 비용은 분할 배낭으로 완화해서 구함 → 실제 비용 이하이므로 안전
          - 기준 미달 카드 집합은 시나리오마다 한꺼번에 낮춘 값(value_failed)을 직접 계산
          - 기준을 채우는 카드의 'T 이전 손실'은 개별 감소량으로 계산 — max 함수는 부분모듈이라 다른 카드를 이미
            낮춘 상태에서의 감소량이 더 크거나 같으므로 안전. 카드별로 몰아주는 결제 집합은 서로 겹치지 않고,
            비용은 미충족 카드 전부를 낮춘 최댓값(lowest) 기준으로 계산해서 시나리오 조합과 상관없이 보수적으로 맞음."""
        n, m, key_of, gain_n = self.n, self.num_cards, self.key_of, self.gain_n
        txs = self.transactions
        pending_set = set(pending)
        pending_pos = {c: j for j, c in enumerate(pending)}
        loss_at = {c: [0.0] * (n - i) for c in pending}   # 결제별: c의 실적조건 혜택을 잃을 때 감소량
        cost_at = {c: [0.0] * (n - i) for c in pending}   # 결제별: 그 결제를 c로 몰 때 비용 (충족 혜택 기준, 작은 쪽)
        cost_before = {c: [0.0] * (n - i) for c in pending}  # 실적 도달 전에 c로 몰 때 비용 (미충족 혜택 기준)
        num_masks = 1 << len(pending)
        value_failed = [value] * num_masks  # mask에 속한 카드들이 기준 미달일 때 Σλ·한도 + Σ_t max 값
        for t in range(i, n):
            row, keys = eff[t], key_of[t]
            vals, degraded = [0.0] * m, [0.0] * m
            for c in range(m):
                factor = 1.0 - lam[keys[c]] if keys[c] >= 0 else 1.0
                vals[c] = row[c] * factor
                # 미충족 카드의 '충족 혜택'을 미충족 혜택으로 낮춘 값
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
            """결제 i..end-1 중 금액 합이 need 이상이 되도록 고를 때 비용 합의 최솟값 (분할 배낭 완화).
            costs_late가 있으면 late_from 이후 결제는 그 비용을 씀(실적을 넘는 결제가 될 수도 있는 구간)."""
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
            return float("inf")  # 이 구간 결제를 다 몰아줘도 못 채움

        def timed_cost(c, need):
            """min_T [T 이전 손실 합 + 0..T 결제로 need를 채우는 비용].
            T를 구간으로 묶어 계산: T ∈ [a, b]면 손실 >= 손실(a), 비용 >= 비용(b) 이므로 그 합은 하한으로 안전."""
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
                # T ∈ [a, b]: a 이전 결제는 확실히 실적 도달 전(미충족 혜택), a..b는 넘는 결제일 수도 있어 충족 기준
                best = min(best, prefix_loss[a - i] + cover_cost(cost_before[c], b + 1, need, cost_at[c], a))
            return best

        needs = [self.gate_values[self.gate_index[c]] - spend[self.gate_index[c]] for c in pending]
        costs = [timed_cost(c, need) for c, need in zip(pending, needs)]

        # 카드마다 몰아주는 결제는 겹칠 수 없으므로, E의 필요 금액 합이 남은 결제 금액 합보다 크면 불가능한 시나리오
        remaining_amount = self.prefix_amount[n] - self.prefix_amount[i]
        best_scenario = None
        for mask in range(1 << len(pending)):
            chosen = [j for j in range(len(pending)) if mask >> j & 1]
            need_sum = sum(needs[j] for j in chosen)
            if need_sum > remaining_amount:
                continue
            cost = sum(costs[j] for j in chosen)
            if len(chosen) >= 2:
                # 두 장 이상 채우는 경우: 필요 금액 합 전체를 서로 겹치지 않는 결제로 채워야 하므로
                # 결제마다 (채우는 카드들 중) 가장 싼 비용으로 합계 need_sum을 덮는 비용도 하한이 됨
                joint = [min(cost_at[pending[j]][idx] for j in chosen) for idx in range(n - i)]
                cost = max(cost, cover_cost(joint, n, need_sum))
            scenario = value_failed[(num_masks - 1) ^ mask] - cost  # E 밖의 카드는 전부 기준 미달로 낮춘 값에서 비용 차감
            if best_scenario is None or scenario > best_scenario:
                best_scenario = scenario
        return int(best_scenario + FLOAT_SLACK)  # E=공집합은 항상 가능하므로 None일 일은 없음

    # ------------------------------------------------------------------ 분기
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
                clamped = min(spend[g] + amount, self.gate_values[g])  # 기준 넘으면 더 쌓여도 결과 동일
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

        # 형제 노드끼리 지배 관계 제거 (완전히 같은 상태면 앞 카드 하나만 남김)
        kept = []
        for child in children:
            if any(_dominates(other[:3], child[:3]) for other in kept):
                continue
            kept = [o for o in kept if not _dominates(child[:3], o[:3])]
            kept.append(child)
        # 스택은 나중에 넣은 것부터 꺼내므로 혜택(·전월실적 누적)이 작은 것부터 push
        # → 혜택 큰 자식을 먼저 탐색해 좋은 incumbent를 빨리 잡음
        kept.sort(key=lambda ch: (ch[0], sum(ch[2])))
        return kept

    def run(self) -> tuple[dict, int]:
        n = self.n
        if n == 0:
            return {}, 0
        init_spend = tuple(0 for _ in self.gated_cards)
        init_lam = tuple(0.0 for _ in self.init_limits)

        best_total, best_path = -1, None
        best_seen = {}  # (깊이, 한도잔액, 전월실적 누적) -> 그 상태에 도달한 최대 혜택 (같은 상태 재탐색 방지)
        path = [0] * n
        nodes = 0
        started = time.perf_counter()
        # 스택 원소: (깊이, 지금까지 혜택, 한도잔액, 전월실적 누적, 직전 결제에 고른 카드, 부모의 라그랑주 승수)
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

            # 가지치기 1: 값싼 상한(결제별 최대 혜택 합)으로 먼저 걸러봄
            if total + self.suffix_best[i] <= best_total:
                continue
            # 가지치기 2: 같은 상태에 이미 같거나 큰 혜택으로 도달한 적 있음
            state = (i, limits, spend)
            if best_seen.get(state, -1) >= total:
                continue
            best_seen[state] = total
            # 가지치기 3: 라그랑주 상한 (incumbent가 아직 없으면 계산 생략하고 첫 리프까지 바로 내려감)
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
        raise ValueError("거래ID가 중복됨 (배정 결과를 거래ID로 구분하므로 고유해야 함)")

    full_assignment = {}
    for month_txs in split_by_month(cleaned).values():
        assignment, searched_total = _search_month(month_txs, cards, table)
        _, checked_total = evaluate_assignment(month_txs, assignment, table)
        assert searched_total == checked_total, "탐색 중 혜택 계산이 common.evaluate_assignment와 불일치"
        full_assignment.update(assignment)

    _, total = evaluate_assignment(cleaned, full_assignment, table)
    return format_result(transactions, full_assignment, cards[0], total)


def _verify_month(transactions: list[dict], cards: list[str], table: dict) -> int:
    """소규모 검증용: 한 달치 결제를 완전탐색 결과와 비교 (결제 건수가 작을 때만 사용)."""
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
