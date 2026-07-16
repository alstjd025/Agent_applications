# EXP-11 — chat deep-overload extension (exp05 계승, queue-bound 영역 관측)

**날짜**: 2026-07-16 · **상태**: running · **브랜치**: `feat/exp07-kv-admission`

## 왜

EXP-10에서 발견한 두-영역 ITL 법칙의 영역 2(큐-바운드: ITL ∝ 대기 토큰
질량, ~3–5ns/tok·step)는 chat에선 관측된 적이 없다 — EXP-05가 raw 용량
(~100 req/s)의 1.67×까지만 밀었기 때문. 같은 붕괴가 chat에서도 rate만 더
올리면 나타나는지 확인해 법칙을 워크로드-불변으로 승격(또는 기각)한다.

## 예측 (사전 등록)

모델: tok/s = running/ITL, running 포화 ≈ 2,900 (KV pool ÷ 요청당 고유
~0.8k tok), ITL(Q) ≈ 125ms + 3.9µs×Q(대기 요청), μ(Q)=running/ITL/225.

| rate (req/s) | 10분 후 Q | ITL | tok/s (peak 대비) |
|---|---|---|---|
| 110 | ~6k | ~150ms | 88% |
| 125 | ~15k | ~185ms | 71% |
| 150 | ~30k | ~245ms | 54% |
| 175 | ~45k | ~310ms | 42% |

기울기가 초선형이면 더 이르게, 큐-무관 구현이면 175까지 평탄(법칙 기각).

## 설계

- 워크로드: `sharegpt_request_level_poisson` (exp05와 동일, num_conversations
  1000 동일 — 순환 배수는 rate에 비례해 커짐을 명기), rate 모드(warmup
  1200rpm×60s 정상 작동), 조건별 cold restart.
- **모델 주의**: exp05는 구 3-70B, 현 스택은 3.1-70B/40960 — 직접 연결이
  안 되므로 **앵커 2점(60, 90 req/s)** 을 같은 조건으로 함께 측정해 3.1-70B
  단일 곡선을 만든다 (3.1-70B chat 무제어 기존점은 exp07 off 50/60/90 5분뿐).
- grid: **60, 90, 110, 125, 150, 175 req/s** (rpm 3600–10500) × **10분**
  (5분이면 큐 축적 부족). λ=200는 부하기 한계로 제외(사용자 결정).
- 부하기: **24 procs × 2048 threads = 49k** in-flight 용량 — λ=175×10분의
  예상 최대 in-flight(~45k)까지 open-loop 유지. chat 붕괴 관측에 수만 연결이
  필요한 것 자체가 장문 워크로드와의 구조적 비대칭(같은 토큰 질량에 27×
  연결).
- 스택: no-timeout gateway(gateway-exp10, 24h) + 64Gi + θ=0 — EXP-10 최종
  재실행과 동일. 드라이버가 3가지 모두 사전 확인.
- 실행: `k8s/exp07/run_exp11_chat.sh {smoke|sweep}` +
  `runner-exp11.template.yaml`. 예상 소요 ~2–2.5h (drain 포함).

## 관측 목표

1. tok/s 하강 곡선 (arrival-앵커 steady + 60s 시간분해)
2. 엔진 ITL vs 대기 토큰 질량 기울기 — exp10의 3–5ns/tok·step과 비교
3. running 2,900 고정 여부 (KV-포화 후 N-pinning 재현)
4. 부하기 무결성: 측정 도착률 = offered, CV≈1, 스레드 클램프 없음

## 결과

(완료 후 기입)
