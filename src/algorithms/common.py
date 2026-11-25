from collections import defaultdict


def clean_transactions(transactions: list[dict]) -> list[dict]:
    result = []
    for tx in transactions:
        amount = int(float(tx["결제금액"]))
        if tx.get("승인취소", "정상") != "정상" or amount <= 0:
            continue
        result.append({
            "거래ID": tx["거래ID"],
            "결제일시": str(tx["결제일시"]),
            "월": str(tx["결제일시"])[:7],
            "업종": tx["업종"],
            "금액": amount,
        })
    result.sort(key=lambda x: x["결제일시"])
    return result


def build_benefit_table(cards: list[str], benefit_db: dict) -> dict:
    """카드/업종별 혜택 목록, 한도키별 월한도, 카드별 전월실적 최소값 구성.
    같은 (카드, 업종) 혜택은 한도를 공유(한도키 공유)하며, 월한도는 그룹 내 최댓값 적용."""
    held = set(cards)
    benefits = {card: defaultdict(list) for card in cards}
    limits = {}
    gates = {}
    for entry in benefit_db["혜택정보"]:
        card = entry["카드ID"]
        if card not in held:
            continue
        category = entry["적용업종"]
        limit_key = f"{card}:{category}"
        tiers = entry.get("전월실적구간")
        parsed = {
            "rate": entry["혜택률또는정액"].get("비율"),
            "flat": entry["혜택률또는정액"].get("정액"),
            "min_spend": min(tiers) if tiers else 0,
            "limit_key": limit_key,
        }
        benefits[card][category].append(parsed)

        monthly_limit = entry.get("월한도")
        if limit_key not in limits:
            limits[limit_key] = monthly_limit
        elif monthly_limit is None or limits[limit_key] is None:
            limits[limit_key] = None
        else:
            limits[limit_key] = max(limits[limit_key], monthly_limit)

        if tiers:
            gates[card] = min(tiers)
    return {"benefits": benefits, "limits": limits, "gates": gates}


def evaluate_assignment(transactions: list[dict], assignment: dict, table: dict) -> tuple[dict, int]:
    """결제목록(날짜순) 전체를 given assignment로 채점. 전월실적은 해당 월 누적(현재 결제 포함) 기준."""
    spend = defaultdict(int)
    remaining_limits = dict(table["limits"])
    benefit_amounts = {}
    for tx in transactions:
        card = assignment[tx["거래ID"]]
        spend[card] += tx["금액"]
        min_spend = table["gates"].get(card, 0)
        eligible = spend[card] >= min_spend

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
            remaining = remaining_limits.get(candidate["limit_key"])
            if remaining is not None:
                amount = min(amount, remaining)
            if amount > best_amount:
                best_amount, best_key = amount, candidate["limit_key"]

        if best_key is not None and remaining_limits.get(best_key) is not None:
            remaining_limits[best_key] -= best_amount
        benefit_amounts[tx["거래ID"]] = best_amount
    return benefit_amounts, sum(benefit_amounts.values())


def split_by_month(transactions: list[dict]) -> dict:
    by_month = defaultdict(list)
    for tx in transactions:
        by_month[tx["월"]].append(tx)
    return by_month


def format_result(transactions: list[dict], assignment: dict, default_card: str, total: int) -> dict:
    result = [
        {"거래ID": tx["거래ID"], "카드ID": assignment.get(tx["거래ID"], default_card)}
        for tx in transactions
    ]
    return {"배정결과": result, "총혜택액": int(total)}
