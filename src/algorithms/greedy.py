

import re
from collections import defaultdict


# 혜택 DB 전처리


def _한도키(혜택: dict) -> str:
    """같은 월 한도를 공유하는 혜택끼리 같은 키를 돌려준다."""
    비고 = 혜택.get("비고") or ""
    m = re.search(r"통합한도 공유\((B\d+)", 비고)
    if m:
        return m.group(1)         
    return 혜택["혜택ID"]          


def _최소실적(혜택: dict) -> int:
    구간 = 혜택.get("전월실적구간")
    if not 구간:
        return 0
    return min(구간)


def 혜택표_만들기(카드목록: list[str], 혜택DB: dict) -> dict:
    """
    {카드ID: {업종: [혜택, ...]}} 형태로 정리하고,
    한도 키별 월한도 {한도키: 월한도(None=무제한)} 도 함께 만든다.
    """
    보유 = set(카드목록)
    표 = {카드: defaultdict(list) for 카드 in 카드목록}
    한도 = {}
    for 혜택 in 혜택DB["혜택정보"]:
        if 혜택["카드ID"] not in 보유:
            continue
        정리 = {
            "혜택ID": 혜택["혜택ID"],
            "비율": 혜택["혜택률또는정액"].get("비율"),
            "정액": 혜택["혜택률또는정액"].get("정액"),
            "최소실적": _최소실적(혜택),
            "한도키": _한도키(혜택),
        }
        표[혜택["카드ID"]][혜택["적용업종"]].append(정리)
        # 같은 한도키가 여러 번 나오면 처음 값(기준 혜택)의 월한도를 쓴다
        한도.setdefault(정리["한도키"], 혜택.get("월한도"))
    return {"혜택": 표, "한도": 한도}


def _원래혜택(혜택: dict, 금액: int) -> int:
    """한도를 생각하지 않은 이 결제 한 건의 혜택액."""
    if 혜택["비율"] is not None:
        return int(금액 * 혜택["비율"])
    if 혜택["정액"] is not None:
        return min(int(혜택["정액"]), 금액)
    return 0


def _이번결제_최대혜택(혜택목록: list, 금액: int, 남은한도: dict):
    """해당 카드·업종의 혜택 중, 남은 한도를 반영했을 때 가장 큰 혜택액과 그 한도키."""
    최고액, 최고키 = 0, None
    for 혜택 in 혜택목록:
        액 = _원래혜택(혜택, 금액)
        남은 = 남은한도.get(혜택["한도키"])
        if 남은 is not None:
            액 = min(액, 남은)
        if 액 > 최고액:
            최고액, 최고키 = 액, 혜택["한도키"]
    return 최고액, 최고키



# 결제 내역 정리


def _결제정리(결제내역: list[dict]) -> list[dict]:
    """취소 건을 빼고, 금액을 정수로 바꾸고, 날짜순으로 정렬한다."""
    결과 = []
    for 순번, 결제 in enumerate(결제내역):
        금액 = int(float(결제["결제금액"]))
        if 결제.get("승인취소", "정상") != "정상" or 금액 <= 0:
            continue
        결과.append({
            "거래ID": 결제["거래ID"],
            "결제일시": str(결제["결제일시"]),
            "월": str(결제["결제일시"])[:7],      # 'YYYY-MM' 단위로 실적·한도를 따로 계산
            "업종": 결제["업종"],
            "금액": 금액,
            "순번": 순번,
        })
    결과.sort(key=lambda x: (x["결제일시"], x["순번"]))
    return 결과


# 배정 결과 평가 (실제 규칙으로 총혜택액 계산)


def 평가(결제목록: list[dict], 배정: dict, 표: dict) -> tuple[dict, int]:
    """
    배정 = {거래ID: 카드ID}
    월별로 카드 실적을 먼저 합산해 실적 미달 혜택은 0으로 처리하고,
    날짜순으로 월 한도를 차감하며 건별 혜택액을 계산한다.
    반환: ({거래ID: 혜택액}, 총혜택액)
    """
    실적 = defaultdict(int)                          # (월, 카드) -> 결제 합계
    for 결제 in 결제목록:
        실적[(결제["월"], 배정[결제["거래ID"]])] += 결제["금액"]

    건별 = {}
    남은한도_월별 = {}
    for 결제 in 결제목록:
        월, 카드 = 결제["월"], 배정[결제["거래ID"]]
        if 월 not in 남은한도_월별:
            남은한도_월별[월] = dict(표["한도"])
        남은한도 = 남은한도_월별[월]

        후보 = [혜택 for 혜택 in 표["혜택"][카드].get(결제["업종"], [])
              if 실적[(월, 카드)] >= 혜택["최소실적"]]
        액, 키 = _이번결제_최대혜택(후보, 결제["금액"], 남은한도)
        if 키 is not None and 남은한도.get(키) is not None:
            남은한도[키] -= 액
        건별[결제["거래ID"]] = 액
    return 건별, sum(건별.values())



# 그리디 배정


def assign(결제내역: list[dict], 카드목록: list[str], 혜택DB: dict) -> dict:
    표 = 혜택표_만들기(카드목록, 혜택DB)
    결제목록 = _결제정리(결제내역)

    배정 = {}
    남은한도_월별 = {}
    for 결제 in 결제목록:
        월 = 결제["월"]
        if 월 not in 남은한도_월별:
            남은한도_월별[월] = dict(표["한도"])
        남은한도 = 남은한도_월별[월]

        # 지금 이 결제에서 혜택이 가장 큰 카드 선택 (동점이면 카드목록 앞쪽 카드)
        최고카드, 최고액, 최고키 = 카드목록[0], -1, None
        for 카드 in 카드목록:
            액, 키 = _이번결제_최대혜택(
                표["혜택"][카드].get(결제["업종"], []), 결제["금액"], 남은한도)
            if 액 > 최고액:
                최고카드, 최고액, 최고키 = 카드, 액, 키

        배정[결제["거래ID"]] = 최고카드
        if 최고키 is not None and 남은한도.get(최고키) is not None:
            남은한도[최고키] -= 최고액

    # 실제 규칙(전월실적 미달 시 혜택 0)으로 다시 계산
    _, 총액 = 평가(결제목록, 배정, 표)

    # INTERFACE.md: 결제내역과 같은 순서/건수로 반환
    # (취소 건은 계산에서 빠지므로 혜택·실적에 영향 없이 첫 번째 카드로 표시)
    배정결과 = [
        {"거래ID": 결제["거래ID"], "카드ID": 배정.get(결제["거래ID"], 카드목록[0])}
        for 결제 in 결제내역
    ]
    return {"배정결과": 배정결과, "총혜택액": int(총액)}



if __name__ == "__main__":
    import csv
    import json
    import time
    from pathlib import Path

    루트 = Path(__file__).resolve().parents[2]

    with open(루트 / "data/cards/card_benefits.json", encoding="utf-8") as f:
        혜택DB = json.load(f)
    with open(루트 / "experiments/카드_조합_실험계획.json", encoding="utf-8") as f:
        조합계획 = json.load(f)

    def csv_읽기(경로):
        with open(경로, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    데이터 = {"실제 93건": csv_읽기(루트 / "data/real/실제결제내역_전처리.csv")}
    for 경로 in sorted((루트 / "data/synthetic").glob("*.csv")):
        데이터[경로.stem] = csv_읽기(경로)

    for 이름, 결제내역 in 데이터.items():
        print(f"\n===== {이름} =====")
        for k in ["2", "3", "4", "8"]:
            총액들, 시작 = [], time.perf_counter()
            for 카드목록 in 조합계획[k]:
                총액들.append(assign(결제내역, 카드목록, 혜택DB)["총혜택액"])
            걸린시간 = (time.perf_counter() - 시작) * 1000
            평균 = sum(총액들) / len(총액들)
            print(f"카드 {k}장 조합 {len(총액들):2d}개 | 평균 총혜택액 {평균:>9,.0f}원 | "
                  f"최소 {min(총액들):>7,}원 | 최대 {max(총액들):>7,}원 | {걸린시간:7.1f}ms")

    예시 = assign(데이터["실제 93건"], 조합계획["3"][0], 혜택DB)
    print(f"\n[예시] 카드 {조합계획['3'][0]} -> 총혜택액 {예시['총혜택액']:,}원")
    for 행 in 예시["배정결과"][:5]:
        print("  ", 행)