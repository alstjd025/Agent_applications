# EXP-07 — KV-occupancy thresholding admission (motivation: 정적 수위 임계의 한계)

**Status**: PREPARED (implementation + assets done; sweep not yet run) ·
**Date**: 2026-07-14 ·
**Branches**: llumnix `feat/kv-admission-threshold`, Agent_applications `feat/exp07-kv-admission`

## 1. 왜 / 가설

KV-tank 유량 분석(ANALYSIS_kv-tank-flow.md)의 대조군 실험. **"현재 KV 점유량(hot)
임계 θ 기반 admission은 suboptimal하다"**를 exp05 시나리오(chat, open-loop)에서
정량화한다. 예상 실패 모드:

1. exp05 실측: 100% attainment 구간(≤50 req/s)의 hot 수위 ≈ **36%**, 60 req/s에서
   66%, 70+에서 87%(pin). SLO를 지키는 θ는 수위를 ~40% 아래에 가둠 → 큰 utilization
   gap. (주의: "빈 HBM 낭비"가 아니라 — 나머지는 idle prefix cache가 hit ~85%를
   만드는 데 쓰임 — **안전하게 더 받을 수 있는 부하를 못 받는 gap**으로 서술)
2. 50→60→70 req/s에서 수위가 36→66→87%로 점프 — θ를 앉힐 중간 수위가 없는 절벽
   → 정적 θ의 비강건성 (θ\*가 rate마다 이동할 것).
3. exp05의 SLO 붕괴는 TBT-위반 지배 — KV 수위는 decode 밀도를 직접 제어하지 못함
   → θ가 완벽해도 TBT SLO는 못 지킴 ("잘못된 변수 제어").

## 2. 구현 (검증 완료 항목)

### Scheduler (llumnix `feat/kv-admission-threshold`, commit 3d55ea7)
- 새 scheduling metric **`kv_cache_usage_ratio`** = CMS `NumUsedGpuTokens /
  NumTotalGpuTokens` — **hot 점유율 [0,1]**, 엔진의 `vllm:kv_cache_usage_perc`와
  동일 축. (기존 `kv_cache_usage_ratio_projected`는 대기 토큰 포함이라 순수
  점유량이 아님 — 가설이 겨냥하는 신호가 아니어서 새로 추가)
- 새 플래그 **`--admission-kv-usage-threshold θ`** (default 0=off): >0이면
  neutral dispatch에 per-instance 하드 필터 추가(`notSkipWhenFallback: true`).
  4개 인스턴스 전부 hot 수위 ≥ θ → `ErrorNoAvailableEndpoint` → scheduler 429.
  **per-instance 의미론**: 여유 인스턴스가 하나라도 있으면 그쪽으로 배치(기존
  argmin selector 불변 — `all_prefills_tokens_num`, exp05와 동일 dispatch).
- 거절 전달 경로(소스 확인): scheduler 429 → gateway `SchedulerClient.Get`
  재시도 루프(기본 5s) → 클라이언트 **503** `no available inference worker`.
  즉시 거절을 위해 gateway args에 `--wait-scheduling-timeout=0s` 추가
  (`patch-gateway-fastfail.sh`; 이미지 불변, args-only).
- 빌드: 호스트에 Go 1.24 설치, cgo sdk 의존 3건 분리(ToolParser any화, lrs
  TokenEncoder 훅) 후 **CGO_ENABLED=0 정적 바이너리** `bin/scheduler-exp07`.
  이미지 빌드 없이 **hostPath로 stock 이미지에 주입** (`patch-scheduler-kvadm.sh`).

### Client (Agent_applications `feat/exp07-kv-admission`)
- `LlumnixCompletionsLLM`: 503/429 + 거절 시그니처 → `LlumnixRejectedError`
  (재시도/논스트리밍 fallback 차단), `_detect_admission_rejection`이
  `is_rejected=True, rejection_reason=KV_THRESHOLD`로 기록. 유닛 테스트 통과.
- 분석: `slo_sliding_window.py` — rejected를 별도 범주로(에러 제외에 안 섞임),
  `attain_pct_offered = attain/(classified+rejected)` 추가.
  `plot_slo_vs_throughput.py` — offered-attainment 라인 + rejection rate 라인
  (거절 있는 런에서만 표시; 무거절 런은 기존 그림과 동일 — exp04로 회귀 확인).

## 3. 실험 설계

고정: 현 라이브 설정 그대로 — **Llama-3.1-70B-Instruct**, TP2×4, max-model-len
40960, `sharegpt_request_level_poisson`, 8 procs × 1024 threads, warmup 20 req/s
× 60s, **5 min/조건**, 조건별 cold restart, rescheduling args exp05와 동일.

> 주의: exp05는 Llama-**3**-70B(32k)로 돌았음 → exp05 수치는 정성 참조만 하고,
> **θ=0(off) 조건 3개를 실험 내 baseline**으로 사용한다.

조건 12개 (`run_exp07.sh sweep`, ~3h):

| rate | θ (hot KV usage) | 세션명 |
|---|---|---|
| 60 req/s (3600rpm) | 0(off), 0.8, 0.6, 0.45, 0.3 | exp07_kvadm_th{0000,0800,0600,0450,0300}_rpm_3600 |
| 90 req/s (5400rpm) | 0(off), 0.8, 0.6, 0.45, 0.3 | 〃 _rpm_5400 |
| 50 req/s (3000rpm) | 0(off), 0.3 | 〃 _rpm_3000 |

θ 그리드 근거: feasible 상한 수위 36% / 붕괴 시작 66% / pin 87% (exp05 실측)
양쪽에 0.3/0.45/0.6/0.8 배치. (3.1-70B에서 수위가 다소 이동 가능 — off 조건
결과를 보고 그리드 ±0.1 조정 여지)

### 스모크 (sweep 전 필수, `run_exp07.sh smoke`)
1. θ=0.0001, 5 req/s, 2조건 중 1: 첫 요청 진행 중 usage>θ → 이후 대부분 즉시
   503 거절. 확인: client `is_rejected/KV_THRESHOLD`, scheduler 로그 "Metric
   based filter applied … kv_cache_usage_ratio", `scheduler_scheduling_failed_total`
   증가, 거절 지연(fast-fail ≤ ~수십 ms).
2. θ=0.99: 거절 0, 정상 완료 (stock-동등성).
3. 부팅 직후 CMS 미수신(MaxFloat32) 거절 폭주가 warmup에 흡수되는지 관찰.

### 측정/그림
- **goodput_offered** (reject=위반; 대표 지표) vs θ, rate별
- attain_admitted vs θ + rejection rate vs θ
- hot KV 수위 밴드 + throughput vs θ (+ prefix hit-ratio 병기)
- TBT p50/p99 vs θ — "θ가 못 지키는 SLO" 논거
- `instance_cms_kv_cache_usage_ratio_projected`/hot 시계열 — 필터가 본 신호 기록

## 4. 결과

(미실행 — sweep 후 기입)
