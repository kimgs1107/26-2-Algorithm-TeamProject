import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

RAW_PATH = "../수원시_추출/tbsh_gyeonggi_day_202603_수원시.csv"
OUT_PATH = "합성_결제내역.csv"
SEED = 42
TARGET_AGE_CODE = 3  # 20-29세
YEAR, MONTH = 2026, 3

# 업종중분류명 -> 5개 확정 업종 매핑
CATEGORY_MAP = {
    "커피/음료": "카페",
    "종합소매점": "편의점·마트",
    "고기요리": "외식", "닭/오리요리": "외식", "별식/퓨전요리": "외식",
    "부페": "외식", "분식": "외식", "양식": "외식", "일식/수산물": "외식",
    "중식": "외식", "패스트푸드": "외식", "한식": "외식", "휴게소/대형업체": "외식",
    "의복/의류": "쇼핑(패션·뷰티)", "패션잡화": "쇼핑(패션·뷰티)",
    "화장품소매": "쇼핑(패션·뷰티)", "인터넷쇼핑": "쇼핑(패션·뷰티)",
    "일반스포츠": "여가·오락", "취미/오락": "여가·오락",
    "공연관람": "여가·오락", "경기관람": "여가·오락",
}
# 제외: 대중교통(F17 교통서비스), 배달(Q09 음식배달서비스) - 집계 구조상 거래건수 적어 분석대상에서 제외
# 그 외 코드(의료, 교육, 공공, 통신 등)는 대학생 소비패턴 분석 범위 밖이라 제외

def hour_to_range(code):
    table = {1:"00-06",2:"07-08",3:"09-10",4:"11-12",5:"13-14",
             6:"15-16",7:"17-18",8:"19-20",9:"21-22",10:"23"}
    return table.get(int(code), "unknown")

def day_to_weekday(code):
    # 1=월 ... 7=일 (pandas weekday: 0=월 ... 6=일)
    return int(code) - 1

def load_filtered():
    df = pd.read_csv(RAW_PATH, encoding="utf-8", engine="python")
    df = df[df["age"] == TARGET_AGE_CODE].copy()
    df["category"] = df["card_tpbuz_nm_2"].map(CATEGORY_MAP)
    total_amt_all = df["amt"].sum()
    df5 = df.dropna(subset=["category"]).copy()
    return df, df5, total_amt_all

def build_stats(df5, total_amt_all):
    g = df5.groupby("category").agg(amt=("amt", "sum"), cnt=("cnt", "sum"))
    g["avg_amt_per_tx"] = g["amt"] / g["cnt"]
    g["share_of_5cat"] = g["amt"] / g["amt"].sum()
    g["share_of_all_20s"] = g["amt"] / total_amt_all
    return g.sort_values("amt", ascending=False)

def build_day_hour_dist(df5):
    dist = {}
    for cat, sub in df5.groupby("category"):
        day_w = sub.groupby("day")["cnt"].sum()
        day_w = (day_w / day_w.sum()).to_dict()
        hour_w = sub.groupby("hour")["cnt"].sum()
        hour_w = (hour_w / hour_w.sum()).to_dict()
        # (가맹점 평균단가, 가중치) 리스트: 실제 집계행별 평균값을 가중 샘플링 풀로 사용
        amt_pool = list(zip((sub["amt"] / sub["cnt"]).tolist(), sub["cnt"].tolist()))
        dist[cat] = {"day": day_w, "hour": hour_w, "amt_pool": amt_pool}
    return dist

def weighted_choice(items_weights, rng):
    items = list(items_weights.keys())
    weights = list(items_weights.values())
    return rng.choices(items, weights=weights, k=1)[0]

def sample_amount(amt_pool, rng):
    vals = [v for v, w in amt_pool]
    weights = [w for v, w in amt_pool]
    base = rng.choices(vals, weights=weights, k=1)[0]
    noise = np.random.lognormal(mean=0, sigma=0.25)  # 건별 금액 변동성 반영
    return max(1000, int(round(base * noise / 100) * 100))  # 100원 단위 보정

def march_dates_by_weekday():
    buckets = {i: [] for i in range(7)}
    d = datetime(YEAR, MONTH, 1)
    while d.month == MONTH:
        buckets[d.weekday()].append(d)
        d += timedelta(days=1)
    return buckets

def generate(n_tx, cat_ratio, dist, seed=SEED):
    rng = random.Random(seed)
    np.random.seed(seed)
    date_buckets = march_dates_by_weekday()
    cats = list(cat_ratio.keys())
    weights = list(cat_ratio.values())
    rows = []
    for i in range(n_tx):
        cat = rng.choices(cats, weights=weights, k=1)[0]
        d = dist[cat]
        weekday = day_to_weekday(weighted_choice(d["day"], rng))
        hour_code = weighted_choice(d["hour"], rng)
        candidates = date_buckets[weekday] if date_buckets[weekday] else sum(date_buckets.values(), [])
        date = rng.choice(candidates)
        amt = sample_amount(d["amt_pool"], rng)
        rows.append({
            "거래ID": f"S{i+1:06d}",
            "결제일시": date.strftime("%Y-%m-%d"),
            "시간대": hour_to_range(hour_code),
            "업종": cat,
            "결제금액": amt,
            "카드ID": "",
            "승인취소": "정상",
        })
    out = pd.DataFrame(rows).sort_values("결제일시").reset_index(drop=True)
    return out

if __name__ == "__main__":
    df_all, df5, total_amt_all = load_filtered()
    stats = build_stats(df5, total_amt_all)
    print("=== 수원시 2026-03 20대 업종별 비중 (확정 5개 업종 기준) ===")
    print(stats.to_string())
    print()
    print(f"5개 업종 합계가 20대 전체 소비(all categories) 중 차지하는 비중: {stats['share_of_all_20s'].sum():.2%}")

    cat_ratio = stats["share_of_5cat"].to_dict()
    dist = build_day_hour_dist(df5)

    for n in [100, 1000]:
        synth = generate(n, cat_ratio, dist)
        path = OUT_PATH.replace(".csv", f"_{n}건.csv")
        synth.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"생성 완료: {path} ({len(synth)}행)")
