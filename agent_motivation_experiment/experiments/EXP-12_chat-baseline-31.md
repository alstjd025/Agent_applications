# EXP-12 — chat 무제어 baseline 재실험 (3.1-70B 단일 모델화)

**날짜**: 2026-07-16 · **상태**: running · **브랜치**: `feat/exp07-kv-admission`

## 왜

아크 전체를 **단일 모델(Llama-3.1-70B)**로 통일하기 위한 chat baseline 재측정.
EXP-05는 구 3-70B(컨텍스트 8k) 시절 데이터인데, EXP-06에서 SWE 체인의 30k
입력 때문에 3.1-70B/40960으로 전환하면서 chat baseline만 구모델로 남아
있었다. EXP-11 앵커 실측으로 두 모델의 chat 거동이 크게 다름이 확인됨
(60 req/s attain 83.8%→23.9%, raw 용량 ~100→~150 req/s) → EXP-05를
legacy로 은퇴시키고 본 실험이 canonical chat baseline이 된다.

## 표준화 결정 (사용자 확정)

- **분석 창 표준 = exp05 관례**: 도착-앵커 **[60s, 340s]** (warmup 60s 제외
  + 본 300s − 끝 20s). 더 긴 run(exp10 8분, exp11 10분)과의 교차 비교는
  `--steady-max-s 360` 재절단으로 같은 창을 사용한다 (과부하는 정상상태가
  없어 창 길이가 다르면 감쇠 궤적의 다른 지점을 평균하게 되므로).
- 프로토콜 = exp05 원형: warmup 20 req/s × 60s + 본 5분, 조건별 cold restart,
  `--post-duration-grace 60`.

## 설계

- 워크로드: `sharegpt_request_level_poisson` (num_conversations 1000, exp05와
  동일), rate 모드.
- grid: **5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120 req/s** (12조건).
  60/90/100은 exp11(10분)과 겹침 → 재현성 교차검증 겸용. 125–175 구간은
  exp11 데이터를 [60,340] 재절단해 단일 곡선에 합류.
- 스택: 3.1-70B 4×TP2, θ=0, no-timeout gateway(gateway-exp10, 24h) + 64Gi —
  exp10/11과 동일. 부하기 24 procs × 2048 threads.
- 실행: `k8s/exp07/run_exp12_chat.sh sweep` (runner-exp11 템플릿 재사용,
  DURMIN=5). 예상 소요 ~2시간.

## 예상 (exp11 앵커 기반)

- tok/s: ~21k에서 포화 평탄 (raw 용량 ~150 req/s라 120까지 초과수요 없음/미미)
- SLO attain: 50 req/s 근처에서 knee (60에서 이미 24%였으므로 3.1-70B의
  SLO-용량은 ~40–55 예상 — 구모델의 60보다 낮음)
- KV: 60 req/s에서 87% → TBT–KV 법칙으로 knee 하향의 근거 설명 가능할 것

## 결과

(완료 후 기입)
