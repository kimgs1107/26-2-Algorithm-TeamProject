from common import clean_transactions, build_benefit_table, evaluate_assignment, split_by_month, format_result

# state = (한도 잔액 튜플, 전월실적 누적 튜플)
# 한도·전월실적 누적값은 1,000원 단위로 반올림해 상태공간을 유한하게 묶는다
# (반올림 전 금액 그대로 쓰면 결제 건수가 늘어날수록 상태 수가 지수적으로 폭증해 DP의 의미가 없어짐)
# 전월실적 누적은 임계값 이상이면 더 쌓여도 결과가 같으므로 임계값에서 clamp
UNIT = 1000
MAX_STATES = 150_000  # 카드 조합이 많거나 결제 건수가 늘면 상태 수가 급격히 커지므로 상한을 둠


def _round(value: int) -> int:
    return round(value / UNIT) * UNIT


def _next_state(limit_state: dict, gate_state: dict, tx: dict, card: str, table: dict):
    min_spend = table["gates"].get(card)
    if min_spend is None:
        eligible = True
    else:
        eligible = (gate_state.get(card, 0) + tx["금액"]) >= min_spend

    best_amount, best_key = 0, None
    for candidate in table["benefits"][card].get(tx["업종"], []):
        if candidate["min_spend"] > 0 and not eligible:
            continue
        if candidate["rate"] is not None:
            amount = int(tx["금액"] * candidate["rate"])
        elif candidate["flat"] is not None:
            amount = min(int(candidate["flat"]), tx["금액"])
        else:
            amount = 0
        remaining = limit_state.get(candidate["limit_key"])
        if remaining is not None:
            amount = min(amount, remaining)
        if amount > best_amount:
            best_amount, best_key = amount, candidate["limit_key"]

    new_limit_state = limit_state
    if best_key is not None and limit_state.get(best_key) is not None:
        new_limit_state = dict(limit_state)
        new_limit_state[best_key] = max(0, _round(limit_state[best_key] - best_amount))

    new_gate_state = gate_state
    if min_spend is not None:
        new_gate_state = dict(gate_state)
        new_gate_state[card] = _round(min(gate_state.get(card, 0) + tx["금액"], min_spend))

    return best_amount, new_limit_state, new_gate_state


def _state_key(limit_state: dict, gate_state: dict) -> tuple:
    return (tuple(sorted(limit_state.items())), tuple(sorted(gate_state.items())))


def _optimize_month(transactions: list[dict], cards: list[str], table: dict) -> tuple[dict, int]:
    init_limits = {k: (v if v is None else _round(v)) for k, v in table["limits"].items()}
    init_gates = {card: 0 for card in cards if card in table["gates"]}
    init_key = _state_key(init_limits, init_gates)

    frontier = {init_key: (0, init_limits, init_gates)}
    backtrace = []  # 층별 {state: (prev_state, card, 거래ID)}

    for tx in transactions:
        next_frontier = {}
        next_backtrace = {}
        for state, (total, limit_state, gate_state) in frontier.items():
            for card in cards:
                amount, new_limit_state, new_gate_state = _next_state(limit_state, gate_state, tx, card, table)
                new_state = _state_key(new_limit_state, new_gate_state)
                new_total = total + amount
                existing = next_frontier.get(new_state)
                if existing is None or existing[0] < new_total:
                    next_frontier[new_state] = (new_total, new_limit_state, new_gate_state)
                    next_backtrace[new_state] = (state, card, tx["거래ID"])
        if len(next_frontier) > MAX_STATES:
            raise ValueError(
                f"DP 상태공간 초과: 결제 {len(transactions)}건 처리 중 상태 수 {len(next_frontier):,}개 "
                f"(상한 {MAX_STATES:,}) — 카드 조합을 줄이거나(2~3장) 결제 건수를 줄여서 다시 시도할 것"
            )
        frontier = next_frontier
        backtrace.append(next_backtrace)

    best_state = max(frontier, key=lambda s: frontier[s][0])
    best_total = frontier[best_state][0]

    assignment = {}
    current = best_state
    for layer in reversed(backtrace):
        prev_state, card, tx_id = layer[current]
        assignment[tx_id] = card
        current = prev_state
    return assignment, best_total


def assign(transactions: list[dict], cards: list[str], benefit_db: dict) -> dict:
    table = build_benefit_table(cards, benefit_db)
    cleaned = clean_transactions(transactions)

    full_assignment = {}
    for month_txs in split_by_month(cleaned).values():
        assignment, _ = _optimize_month(month_txs, cards, table)
        full_assignment.update(assignment)

    # 보고용 총혜택액은 반올림 없이 실제 규칙으로 다시 계산(배정 자체는 DP 결과 그대로 사용)
    _, total = evaluate_assignment(cleaned, full_assignment, table)
    return format_result(transactions, full_assignment, cards[0], total)


if __name__ == "__main__":
    import csv
    import json
    import time
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]

    with open(root / "data/cards/card_benefits.json", encoding="utf-8") as f:
        benefit_db = json.load(f)
    with open(root / "data/real/실제결제내역_전처리.csv", encoding="utf-8-sig") as f:
        transactions = list(csv.DictReader(f))

    for n in (10, 30, len(transactions)):
        sample = transactions[:n]
        start = time.perf_counter()
        result = assign(sample, ["CARD01", "CARD02"], benefit_db)
        print(f"N={n:3d} {(time.perf_counter()-start)*1000:8.1f}ms, 총혜택액 {result['총혜택액']:,}원")

    # 카드 조합이 늘면(3장+) 상태공간이 커져 상한에 걸릴 수 있음(의도된 안전장치)
    try:
        assign(transactions, ["CARD02", "CARD07", "CARD08"], benefit_db)
    except ValueError as e:
        print(f"\n[3장 조합 예시] {e}")
