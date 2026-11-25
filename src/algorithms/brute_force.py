import itertools

from common import clean_transactions, build_benefit_table, evaluate_assignment, split_by_month, format_result

# 월·한도·전월실적은 월 단위로만 영향을 주므로 월별로 완전탐색을 분리해 돌린다
# (카드 수)^(해당 월 결제 건수) 만큼 조합을 다 평가하므로 월별 건수가 크면 비현실적으로 느려진다
MAX_COMBINATIONS = 2_000_000


def _search_month(transactions: list[dict], cards: list[str], table: dict) -> tuple[dict, int]:
    combinations = len(cards) ** len(transactions)
    if combinations > MAX_COMBINATIONS:
        raise ValueError(
            f"완전탐색 불가 범위: 카드 {len(cards)}장 x 결제 {len(transactions)}건 = {combinations:,}가지 "
            f"(상한 {MAX_COMBINATIONS:,}). 검증용으로 더 작은 결제 건수 부분집합을 사용할 것"
        )
    best_total, best_assignment = -1, None
    for combo in itertools.product(cards, repeat=len(transactions)):
        assignment = {tx["거래ID"]: card for tx, card in zip(transactions, combo)}
        _, total = evaluate_assignment(transactions, assignment, table)
        if total > best_total:
            best_total, best_assignment = total, assignment
    return best_assignment, best_total


def assign(transactions: list[dict], cards: list[str], benefit_db: dict) -> dict:
    table = build_benefit_table(cards, benefit_db)
    cleaned = clean_transactions(transactions)

    full_assignment = {}
    for month_txs in split_by_month(cleaned).values():
        assignment, _ = _search_month(month_txs, cards, table)
        full_assignment.update(assignment)

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

    small_sample = transactions[:10]
    start = time.perf_counter()
    result = assign(small_sample, ["CARD02", "CARD07", "CARD08"], benefit_db)
    print(f"{(time.perf_counter()-start)*1000:.1f}ms, 총혜택액 {result['총혜택액']:,}원")
    for row in result["배정결과"]:
        print("  ", row)
