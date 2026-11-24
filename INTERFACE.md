# 공통 입출력 인터페이스

## 입력

- `결제내역`: `list[dict]`, 각 원소는 `{"거래ID": str, "결제일시": str, "업종": str, "결제금액": int}`
- `카드목록`: `list[str]`, 보유 카드 ID 리스트 (`data/cards/card_benefits.json`의 카드ID, 부분조합 실험 시 `experiments/카드_조합_실험계획.json`에서 꺼내 씀)
- `혜택DB`: `dict`, `data/cards/card_benefits.json`를 그대로 로드한 값

## 출력

```python
{
    "배정결과": [{"거래ID": str, "카드ID": str}, ...],  # 결제내역과 같은 순서/건수
    "총혜택액": int,
}
```

## 함수 시그니처

```python
def assign(결제내역: list[dict], 카드목록: list[str], 혜택DB: dict) -> dict:
    ...
```

- 모든 알고리즘(`greedy`, `dp`, `branch_and_bound`, `genetic`, `brute_force`)은 이 시그니처를 그대로 따름
- 전월실적·월한도 등 혜택 계산 로직은 `혜택DB["혜택정보"]`의 `전월실적구간`/`월한도` 필드를 참고
- 실행시간·메모리 측정은 이 함수 밖(실험 스크립트)에서 담당하므로 `assign` 내부에 넣지 않음
