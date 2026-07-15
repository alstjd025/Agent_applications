# EXP-10 — SWE open-loop request-level replay λ sweep (admission 없음)

**날짜**: 2026-07-15 · **상태**: running · **브랜치**: `feat/exp07-kv-admission`

## 왜 (motivation)

EXP-06/09의 SWE 워크로드는 **job-level release (closed-loop)**: 콜 n+1은 콜 n
완료 + tool-delay 후에야 발행되고, 거절/지연 1건이 체인의 미래 콜 발행 자체를
줄인다 (EXP-09 실측: mid-chain 거절 1건 = 체인 즉시 사망, 후속 콜 4.6× 억압).
즉 시스템이 받는 도착 스트림이 시스템 자신의 상태에 되먹임되는 구조라,
"무제어 붕괴 곡선"이 순수 수요 곡선이 아니다.

EXP-10은 같은 콜 내용(EXP-06/09와 동일한 SWE agent 프롬프트)을 **open-loop
request-level**로 릴리즈해 되먹임을 제거한다 — EXP-05(chat open-loop)의 SWE
버전. 목적:

1. chain 되먹임 없는 상태의 진짜 용량·붕괴 지점 (용량 추정: prefill 기준
   ~6 req/s, KV 체류 기준 ~13 req/s 사이)
2. TBT–KV 선형 법칙(ANALYSIS_why-not-full-kv)의 3번째 독립 검증점
   (chat open-loop, SWE closed-loop에 이어 SWE open-loop)
3. EXP-06/09 곡선과의 비교축 확보 — call 도착률(calls/s)로 환산해 오버레이

admission은 걸지 않는다 (θ=0, stock 의미론). θ sweep은 보류된 원안(EXP-09
문서 말미)에서 제외됐다.

## 예상

- attain 곡선은 chat(EXP-05)형 절벽을 보이되, 5.5k tok/req prefill 부하 때문에
  붕괴점은 훨씬 낮은 λ (6–13 req/s 사이 예상).
- TBT–KV 기울기 ~20.6 ms/Mtok이 유지되면 법칙의 워크로드·릴리즈방식 불변성 강화.

## 설계 (사용자 확정: 녹화 60분 / λ 11점 1–20 / 조건 8분)

### E1. Transcript 녹화

replay workload(`codingagent_request_level_poisson`)는 transcript JSONL을
literal replay하므로 녹화가 선행된다.

- `swe_bench_coding_tool_delay`, 0.5 jobs/s × 60분, θ=off, chain 4–12,
  warmup 0.25 jobs/s × 60s, cold restart 선행.
- **LOAD_PROCS=1**: 여러 proc이 같은 transcript 파일에 append하면 ~22KB
  JSONL 라인이 인터리브로 깨질 수 있어 단일 proc으로 녹화.
- 기대 ~10k+ 라인 (성공 콜만 기록). 산출:
  `results/exp10_transcript/transcript_swe_calls.jsonl` (~300MB, 커밋 금지).
- pool 크기 근거: replay pool은 소진 시 동일 바이트로 순환 → 순환 ≥2부터
  prefix-cache hit가 비현실적으로 올라 prefill 부하가 증발. 최고 λ 조건
  (20 req/s × 8분 ≈ 9.6k 요청)에서도 순환 ≤~1이 되게 8k+ 라인 확보.

### E2. λ sweep

- λ = 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20 req/s (11조건) × 8분,
  조건별 cold restart, warmup 1 req/s(=60 rpm) × 60s.
- `--disable-timeouts`: client abort 전부 끔 (τ-timeout 포함) — 느린 요청도
  죽이지 않고 측정 (EXP-05 chat과 동일 철학, absolute-SLO 분석).
- MP 8 procs: 각 worker가 transcript를 `[shard::8]`로 나눠 갖고 독립 순환 →
  worker 간 중복 재생 없음, worker당 순환 배수는 전역과 동일.
- 실행: `k8s/exp07/run_exp10_replay.sh {record|smoke|sweep}` +
  `runner-exp10{,-record}.template.yaml`. 드라이버가 시작 전 scheduler
  admission off(θ=0)를 강제 확인.

### 분석

1. request SLO(TTFT≤5s & meanTBT≤50ms, steady [60s, dur−20s]) attain vs λ
   — exp05 파이프라인 (`slo_sliding_window.py` 등). rejected 범주는 0이어야 함.
2. λ별 ITL CDF — `plot_itl_cdf_rates.py` glob 교체.
3. 엔진 상태(KV 수위/running/prefill) + TBT–KV fit (`analyze_tbt_drivers.py`).
4. EXP-06/09 오버레이: jobs/s 축 직접 비교 불가 → 각 run의 **측정된 call
   도착률(calls/s)** 로 환산해 비교.
5. 조건별 replay_index 분포 로그 → prefix-hit 왜곡 사후 검증.

## 세팅 요약

| 항목 | 값 |
|---|---|
| 모델/서빙 | Llama-3.1-70B-Instruct, 4×TP2, max-model-len 40960, util 0.9 |
| admission | 없음 (θ=0; custom scheduler 바이너리 유지, stock 의미론) |
| gateway | stock (wait-scheduling-timeout 기본값, fast-fail 해제) |
| release | request-level open-loop Poisson (λ = req/s) |
| 측정 | 8분/조건, steady [60s, 460s], 조건별 cold restart |
| 클라 timeout | 전부 비활성 (`--disable-timeouts`) |

## 결과

(완료 후 기입)
