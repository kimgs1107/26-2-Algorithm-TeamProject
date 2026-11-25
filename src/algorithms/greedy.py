from common import clean_transactions, build_benefit_table, evaluate_assignment, split_by_month, format_result

# 매 결제마다 "지금 당장 혜택이 가장 큰 카드"를 고르는 탐욕적 선택
# (전월실적 누적은 cumulative-to-date 기준: 미래 결제까지 보고 판단하지 않음)


def _best_card(tx: dict, cards: list[str], gate_state: dict, limit_state: dict, table: dict):
    best_card, best_amount, best_key = cards[0], -1, None
    for card in cards:
        min_spend = table["gates"].get(card)
        if min_spend is None:
            eligible = True
        else:
            eligible = (gate_state.get(card, 0) + tx["금액"]) >= min_spend

        amount, key = 0, None
        for candidate in table["benefits"][card].get(tx["업종"], []):
            if candidate["min_spend"] > 0 and not eligible:
                continue
            if candidate["rate"] is not None:
                a = int(tx["금액"] * candidate["rate"])
            elif candidate["flat"] is not None:
                a = min(int(candidate["flat"]), tx["금액"])
            else:
                a = 0
            remaining = limit_state.get(candidate["limit_key"])
            if remaining is not None:
                a = min(a, remaining)
            if a > amount:
                amount, key = a, candidate["limit_key"]

        if amount > best_amount:
            best_card, best_amount, best_key = card, amount, key
    return best_card, best_amount, best_key


def _assign_month(transactions: list[dict], cards: list[str], table: dict) -> dict:
    limit_state = dict(table["limits"])
    gate_state = {card: 0 for card in cards if card in table["gates"]}
    assignment = {}
    for tx in transactions:
        card, amount, key = _best_card(tx, cards, gate_state, limit_state, table)
        assignment[tx["거래ID"]] = card
        if key is not None and limit_state.get(key) is not None:
            limit_state[key] -= amount
        if card in gate_state:
            gate_state[card] += tx["금액"]
    return assignment


def assign(transactions: list[dict], cards: list[str], benefit_db: dict) -> dict:
    table = build_benefit_table(cards, benefit_db)
    cleaned = clean_transactions(transactions)

    full_assignment = {}
    for month_txs in split_by_month(cleaned).values():
        full_assignment.update(_assign_month(month_txs, cards, table))

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
    with open(root / "experiments/카드_조합_실험계획.json", encoding="utf-8") as f:
        combo_plan = json.load(f)

    def read_csv(path):
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    datasets = {"실제 93건": read_csv(root / "data/real/실제결제내역_전처리.csv")}
    for path in sorted((root / "data/synthetic").glob("*.csv")):
        datasets[path.stem] = read_csv(path)

    for name, transactions in datasets.items():
        print(f"\n===== {name} =====")
        for k in ["2", "3", "4", "8"]:
            totals, start = [], time.perf_counter()
            for cards in combo_plan[k]:
                totals.append(assign(transactions, cards, benefit_db)["총혜택액"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            avg = sum(totals) / len(totals)
            print(f"카드 {k}장 조합 {len(totals):2d}개 | 평균 총혜택액 {avg:>9,.0f}원 | "
                  f"최소 {min(totals):>7,}원 | 최대 {max(totals):>7,}원 | {elapsed_ms:7.1f}ms")

    example = assign(datasets["실제 93건"], combo_plan["3"][0], benefit_db)
    print(f"\n[예시] 카드 {combo_plan['3'][0]} -> 총혜택액 {example['총혜택액']:,}원")
    for row in example["배정결과"][:5]:
        print("  ", row)
