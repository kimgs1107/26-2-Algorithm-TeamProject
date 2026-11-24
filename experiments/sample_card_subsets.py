import itertools
import json
import random

SEED = 42
OUT_PATH = "카드_조합_실험계획.json"

# 카드 혜택 DB 확정 전 임시 ID. DB 완성되면 실제 카드ID로 교체
CARD_POOL = [f"CARD{i:02d}" for i in range(1, 11)]

# 2~4장: 현실적인 대학생 보유 수준 -> 조합별 근사율 분산까지 확인
# 5/8/10장: 기존 계획 참고용 비교점(상한선)으로 유지
K_SAMPLE_COUNT = {2: 20, 3: 20, 4: 20, 5: 10, 8: 5, 10: 1}

def all_combos(pool, k):
    return list(itertools.combinations(pool, k))

def sample_subsets(pool, k, n_samples, rng):
    combos = all_combos(pool, k)
    if len(combos) <= n_samples:
        return [list(c) for c in combos]
    picked = rng.sample(combos, n_samples)
    return [list(c) for c in picked]

def build_plan(pool=CARD_POOL, k_sample_count=K_SAMPLE_COUNT, seed=SEED):
    rng = random.Random(seed)
    plan = {}
    for k, n_samples in sorted(k_sample_count.items()):
        plan[k] = sample_subsets(pool, k, n_samples, rng)
    return plan

if __name__ == "__main__":
    plan = build_plan()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    for k, subsets in plan.items():
        print(f"k={k}: 조합 {len(subsets)}개 생성")
    print(f"저장 완료: {OUT_PATH}")
