# EXP-169 (분석, 새 run 없음): 왜 FluidServe가 기준선보다 나은가 — 클러스터 용량을 어디까지 쓰는가, 그리고 덜 쓰거나 더 쓰는 원인

**날짜**: 2026-11-20. **사용자 질문**: "우리 시스템이 저 baseline들보다 뛰어난 이유를 파악해볼래? 우리가 Cluster의 capacity를 거의 최대로 쥐어 짠다는 맥락에서",
"llm-d는 예측기 P50을 한 것도 있잖아 … 모든 실험 경우들에 대해서 세 클래스의 mean TTFT, TBT, tail latency까지 … 시각화 … 한계보다 못쓰거나 더 써버린다면 그 원인을 명확히 규명해."

**데이터**: EXP-161(섞인 trace, v0.5·v0.6), EXP-164(단일 클래스, v0.5·v0.6), EXP-165(PolyServe·Llumnix SLO·llm-d p90), EXP-166·167(llm-d p50, 섞인 trace llm-d), EXP-168 r1 PolyServe(섞인 trace).
섞인 trace의 Llumnix SLO 두 arm은 EXP-168이 끝나면 넣는다. 세션이 다른 비교가 섞여 있다(EXP-161 11-16, EXP-164 11-17~18, EXP-165 11-18~19, EXP-166~168 11-19~20).

**스크립트·그림** (안쪽 저장소 `analysis_scripts/capacity/`, 출력 `results/aggregate_analysis/capacity/`):
- `engine_work.py` — run마다 엔진의 vLLM 지표(5~55분): 생성 토큰/s, prompt 토큰/s, prefix 적중, 엔진당 동시 처리·대기, KV, 엔진이 잰 토큰 간격, preemption, 그리고 ladder95 goodput과 그 비율. 결과 `/home/nxclab/scratch/capacity/engine_work.tsv`.
- `root_cause.py memory|pace|engines` — PolyServe 메모리 판정의 분해, 판정 시점의 예측 step 대 실측, 엔진별·5분별 토큰 간격.
- `fig_breakdown.py` — `breakdown_<trace>.png/.csv`(클래스별 TTFT·TBT 평균 막대 + P99 수염, 도착별 결과), `operating_point.png/.csv`(정책별 5분 구간 토큰 간격·동시 처리).

## 1. 한계는 어디인가

엔진 하나의 출력 처리량 ≈ (동시 처리 요청 수) ÷ (토큰 하나 걸리는 시간). 동시 처리가 늘면 step이 길어지므로, SLO를 지키며 낼 수 있는 최대 처리량은 **토큰 간격이 그 엔진에서 처리 중인 가장 빡빡한 클래스의
budget에 닿을 때까지**(또는 KV가 물리 용량에 닿을 때까지) 받는 지점이다. 거기서 더 받으면 받은 요청이 늦고, 덜 받으면 처리량이 남는다.

**FluidServe v0.6의 운전점** (5분 구간, 엔진이 잰 토큰 간격 / budget): chat 2.5배 엔진 넷 모두 45.6~50.5 ms / 50(구간 안 엔진 간 차이 평균 1.7 ms); swe만 77~82 / 75;
섞인 trace에서는 chat을 가진 엔진이 45~50 ms, swe·deepresearch만 가진 엔진이 70~77 ms로 **클래스마다 다른 budget에 맞춰 갈라진다**. deepresearch만은 KV가 한계(KV 0.99, 토큰 간격 73 ms / 100).

## 2. 정책별 결과 (ladder95, 2회 평균; 엔진 출력 = 엔진이 만든 출력 토큰/s, 마감 비율 = goodput ÷ 엔진 출력)

| trace | 정책 | 모든 도착 | admitted | goodput | 엔진 출력 | 마감 비율 | 엔진당 동시 처리 | 엔진 토큰 간격 (ms) | KV 평균 |
|---|---|---|---|---|---|---|---|---|---|
| chat 2.5배 (50 ms) | v0.6 | 68.16 | 99.94 | 19,862 | 20,548 | 0.97 | 253 | 49.0 | 0.37 |
| | llm-d p90 | 61.19 | 99.47 | 18,119 | 18,749 | 0.97 | 208 | 44.2 | 0.32 |
| | llm-d p50 | 54.83 | 95.97 | 16,172 | 17,163 | 0.94 | 208 | 48.2 | 0.32 |
| | PolyServe | 19.94 | 29.09 | 7,975 | 20,705 | 0.39 | 371 | 71.4 | 0.58 |
| | Llumnix SLO | 21.48 | 31.34 | 7,652 | 20,629 | 0.37 | 401 | 77.6 | 0.65 |
| swe만 (75 ms) | v0.6 | 54.01 | 97.68 | 7,342 | 7,739 | 0.95 | 155 | 79.1 | 0.66 |
| | llm-d p90 | 42.53 | 93.64 | 6,040 | 6,326 | 0.95 | 102 | 63.6 | 0.48 |
| | PolyServe | 42.05 | 99.78 | 5,688 | 5,751 | 0.99 | 77 | 52.9 | 0.38 |
| | Llumnix SLO | 2.65 | 4.83 | 389 | 7,627 | 0.05 | 225 | 117.3 | 0.99 |
| deepresearch만 (100 ms) | v0.6 | 38.09 | 97.35 | 10,411 | 10,561 | 0.99 | 193 | 72.9 | 0.99 |
| | PolyServe | 31.66 | 100.00 | 8,391 | 8,454 | 0.99 | 104 | 49.1 | 0.60 |
| | llm-d p90 | 15.06 | 86.70 | 4,547 | 4,764 | 0.96 | 75 | 63.9 | 0.49 |
| | Llumnix SLO | 0.46 | 0.95 | 5,294 | 12,904 | 0.41 | 251 | 77.6 | 1.00 |
| 섞인 논문 trace | v0.6 | 80.69 | 99.9 | 11,764 | 11,975 | 0.98 | 167 | 55.6 | 0.57 |
| | llm-d p50 | 63.07 | 94.16 | 9,396 | 10,122 | 0.93 | 124 | 48.7 | 0.44 |
| | llm-d p90 | 56.76 | 98.71 | 8,950 | 9,400 | 0.95 | 101 | 42.6 | 0.38 |
| | PolyServe (r1만) | | | | 11,267 | | 159 | 56.3 | 0.44 |

클래스별 TTFT·TBT(평균, P99)와 결과는 `breakdown_<trace>.png/.csv`.

## 3. 원인 — 기준선마다, 그 정책 자신의 기록으로

### 3.1 덜 쓰는 원인

**PolyServe: 메모리 판정이 논리 토큰 수와 기대 출력 전체로 KV를 계산한다.** 판정은 `projected = 인스턴스의 decode 토큰 수 + 새 prompt + (decode batch + 1) × tier 기대 출력 > 용량`이면 거절이다
(`polyserve.go:126-139, 570-573`). 1초마다 스케줄러가 기록한 인스턴스별 값으로 분해하면(`root_cause.py memory`, 각 trace r1; 용량 584,928 토큰/인스턴스):

| trace | 물리 KV (엔진) | 논리 decode 토큰 (prefix 공유를 풀어 센 수) | 증가 항 | 가장 큰 인스턴스의 projected / 용량 |
|---|---|---|---|---|
| swe만 | **37.7 %** | **91.5 %** (물리의 2.43배) | 6.6 % | 1.00 (중앙값과 p90 모두) |
| deepresearch만 | 60.0 % | 81.4 % (1.36배) | 17.8 % | 1.00 |
| chat 2.5배 | 61.8 % | 69.2 % (1.13배) | **28.9 %** | 1.00 |

projected는 run 내내 정확히 용량에 붙어 있다 — 메모리 판정이 받는 양을 정한다. swe는 prompt가 길고(약 5.6k 토큰) 앞부분을 길게 공유해 **논리 토큰이 물리 KV의 2.4배**이므로, 물리 KV 38 %에서 "가득 찼다"고 판정한다.
그 결과 swe만에서 엔진당 동시 처리 77·토큰 간격 53 ms로 budget 75 ms의 70 %에서 멈춘다. chat에서는 증가 항(처리 중 요청마다 이미 만든 토큰과 상관없이 기대 출력 428토큰 전체)이 크다. EXP-127(8 인스턴스)과 같은 원인이고,
2026-09-11 사용자 결정으로 논문 충실성 때문에 고치지 않았다.

**llm-d (p90): 예측기가 TTFT·TPOT의 90번째 백분위수를 예측하고 그것을 SLO와 여유 없이 비교한다.** (`prediction.go:149-177`, `LATENCY_QUANTILE_ALPHA` 기본 0.9). EPP 예측 평균은 실측보다 TPOT 8~13 %,
TTFT 16~32 % 높다(chat·섞인 trace·swe만). 그래서 엔진 토큰 간격이 chat 41~47 ms(budget 50), swe 62~67 ms(75), 섞인 trace 37~48 ms에서 멈춘다.

**llm-d (섞인 trace): 모든 엔진이 chat의 50 ms에 묶인다.** llm-d는 chat을 네 대 모두에 퍼뜨리고(논문 그림 4, `evidence.md` §4.5), 엔진을 판정할 때 그 엔진에서 처리 중인 요청 중 가장 빡빡한 TPOT SLO(`podMinTPOTSLO`)를 쓴다
(`prediction.go:162-168`). 그래서 swe·deepresearch도 50 ms 기준(여기에 p90 여유)으로 판정되어 엔진 넷이 모두 36~52 ms에서 돈다. v0.6은 chat을 일부 엔진에 모으고 나머지 엔진을 75 ms 근처까지 쓴다
(5분 구간 엔진 간 차이 평균 v0.6 21.7 ms 대 llm-d 6.8 ms). **섞인 trace에서 v0.6의 이득의 상당 부분은 클래스 분리가 swe·deepresearch 엔진을 느슨한 budget까지 쓰게 하는 데서 온다** — 엔진당 동시 처리 167 대 101.

**llm-d (deepresearch만): 한 엔진에 몰리고, 그 한 엔진 때문에 클러스터 전체가 거절한다.** 두 메커니즘이 겹친다(r1, 3분 구간):
- 경로 선택: latency scorer의 기본 전략이 `least` — SLO 안이라고 예측된 엔진 중 **여유가 가장 적은** 엔진을 고른다("bin-packing to preserve capacity on other endpoints", `scorer/latency/plugin.go:44-56, 89`).
  deepresearch의 budget은 느슨하고(10 s / 100 ms) 예측기는 KV·preemption을 모르므로, 가득 찬 엔진도 계속 "SLO 안"이고 계속 그 엔진이 골라진다. r1 처음 21분 동안 요청의 81~97 %가 엔진 8001로 갔다.
  (prefix 친화 필터가 원인이라는 앞의 추정은 틀렸다 — 그 필터는 deepresearch 결정의 98 % 이상에서 "일치 없음"이었다.)
- admission: 흐름 제어가 꺼져 있어 legacy 포화 검사가 돈다. 풀 포화도 = 엔진별 max(대기열 ÷ 5, KV ÷ 0.8)의 **평균**이고 1.0 이상이면 sheddable 요청을 429로 거절한다(`saturationdetector/utilization/detector.go:150-165`,
  `requestcontrol/admission.go:75`). 대기열 비율에 상한이 없어서, 엔진 8001 하나의 대기 24~27건·KV 100 %가 나머지 세 엔진의 KV 0~2 %를 덮고 평균을 1.18~1.34로 만든다. 그래서 거절 86 %(전부 429)인 동안 세 엔진이 거의 논다.
  r2는 처음 몇 분만 몰리고 고르게 퍼져 모든 도착 18.3(r1 11.8)이었다 — 이 몰림이 run마다 생기는지가 llm-d의 큰 반복 편차의 원인이다.

### 3.2 더 쓰는 원인

**판정 시점의 step 예측이 실측의 55~67 %다** (`root_cause.py pace`, 판정 로그의 예측값 평균 대 같은 5분 구간 엔진 토큰 간격, 모든 구간):

| run | 예측 | 실측 | 예측 / 실측 |
|---|---|---|---|
| PolyServe chat 2.5배 (`polyserve_iter_max`) | 41~43 ms | 66~75 ms | 0.58~0.62 |
| PolyServe 섞인 trace | 35~41 ms | 49~61 ms | 0.65~0.71 |
| Llumnix SLO chat 2.5배 (`predicted_tpot`) | 36~45 ms | 63~87 ms | 0.51~0.57 |
| Llumnix SLO swe만 | 65~67 ms | 117~118 ms | 0.56 |
| Llumnix SLO deepresearch만 | 59~62 ms | 75~81 ms | 0.77 |

- **PolyServe**는 정상상태 iteration 추정에서 prefill을 뺀다(논문 §4.5, `--polyserve-steady-state-ignores-prefill`). chat처럼 prefill이 계속 섞이면 실제 iteration에는 prefill chunk가 들어 있으므로 예측이 실측의 약 60 %다.
  그래서 예측이 50 ms 아래인 동안 계속 받아 chat 엔진이 66~75 ms로 돈다. 섞인 trace에서 PolyServe는 chat 두 엔진을 51~69 ms로 넘기고(이 쪽은 더 쓴다), swe·deepresearch 엔진은 메모리 판정으로 43~51 ms에서 멈춘다(이 쪽은 덜 쓴다) — 두 원인이 tier별로 따로 나타난다.
- **Llumnix SLO**는 새 요청 하나를 넣었을 때 다음 한 step의 decode 시간을 프로파일에서 예측해 0.9 × 자기 TPOT과 비교한다. prefill 간섭도, 받은 뒤 batch와 KV가 커지는 것도 들어가지 않고, KV를 보는 판정이 없다.
  그래서 swe만·deepresearch만에서 KV가 99~100 %까지 차고 preemption이 9천·2만 회 나며 받은 요청이 첫 토큰으로 실패한다.
- 한 번 budget을 넘으면 요청이 오래 머물러 동시 처리가 늘고(chat 370~450) step이 더 길어진다. 같은 도착을 같은 비율로 받으면서도(chat 2.5배 거절률 31 % 셋 다) 엔진 출력은 같고(20.6~20.8k) 마감 비율만 0.37~0.39로 떨어진다.

**llm-d p50: 예측이 실측보다 조금 낮고, 엔진을 SLO 바로 아래로 채운다.** EPP 예측 평균이 실측보다 TPOT 3~9 %, TTFT 4~14 % 낮다. 거기에 `least` 전략이 엔진을 예측 SLO에 붙여 채우므로 실제 간격이 budget을 넘는 요청이
chat에서 36~37 %(p90 12 %)다. 운전점은 v0.6과 비슷하지만(chat 엔진 토큰 간격 48 대 49 ms) 엔진 간·시간 간 차이가 크다(구간 안 엔진 간 차이 평균 3.3 대 1.7 ms, 최대 54 ms).

### 3.3 FluidServe가 한계에 붙는 이유 (같은 기록으로)

- 판정하는 양이 **인스턴스의 평균 iteration 시간**이고, prefill 간섭 지연을 넣고(prefill 간섭 ÷ h), 인스턴스마다 관측 step으로 보정한다 — EXP-164에서 iteration 예측/실측 1.001. 기준선의 0.51~0.77과 대조된다.
- 판정선이 그 인스턴스에서 처리 중인 가장 빡빡한 클래스 budget의 1.0배이고, 처리 중 요청의 남은 허용치를 넘기지 않는 조건이 같이 걸린다. 그래서 엔진 넷이 같은 budget에 고르게 붙는다(chat 엔진 간 차이 1.7 ms).
- KV는 끝점 투영과 물리 KV × 0.95로 판정해(prefix 공유를 푼 논리 토큰이 아니다) deepresearch만에서 KV 0.99까지 쓴다.
- 클래스 분리(affinity)로 chat이 없는 엔진의 판정선이 75·100 ms가 되어, 섞인 trace에서 swe·deepresearch 엔진이 그 budget 가까이까지 쓴다.

### 3.4 v0.6의 남은 약점 (같은 그림에서)

- swe만: 엔진 토큰 간격이 budget을 4~7 ms 넘는다(77~82 ms; 마감 비율 0.95). 판정 시점 추정이 받은 뒤의 실제보다 낮은 문제(EXP-164).
- deepresearch만: KV가 한계여서 preemption 1.2만 회. 토큰 간격이 73 ms로 budget 100 안이라 대부분 흡수되지만, KV 판정이 너무 많이 받는다(EXP-164).

## 4. 한계

- 세션이 다른 비교가 섞여 있다(세션 간 최대 4.6점). 섞인 trace의 v0.6은 EXP-161(11-16)이고 다시 돌리지 않는다(사용자 결정).
- `pace`의 예측값은 판정 로그에 찍힌 "골라진 인스턴스의 예측"의 평균이고 실측은 같은 5분 구간 클러스터 전체 토큰 간격이다(인스턴스-포트 대응 없이). 예측/실측 비율이 모든 구간에서 같은 방향이라 결론에는 영향이 없다고 본다.
- llm-d deepresearch 몰림의 처음 시작(어느 엔진이 왜 처음 골라지는가)은 기록이 없어 확인하지 않았다. 몰림을 유지하는 메커니즘(`least` 전략 + 예측기가 KV를 모름)과 거절을 만드는 메커니즘(포화 평균)은 코드와 지표로 확인했다.
- Llumnix SLO의 `predicted_tpot`이 decode만의 한 step이라는 것은 판정 로그의 값(실측의 51~77 %)과 필터 설명(`scheduling_policy_registry.go:207-223`)에서 본 것이고 예측 함수 내부는 다시 읽지 않았다.

## 5. 섞인 trace와 단일 클래스 trace 전체 — 정책별 원인 (2026-11-20, 하위 분석 넷, 사용자 지시 "섞인 트레이스와 섞이지 않은 트레이스 모두 … 멀티에이전트화")

스크립트·출력: `/home/nxclab/scratch/exp169/{mixed_ps_slo,mixed_llmd,fluidserve,single_class}/`(각 `NOTES.md`/`NOTES.txt`). 섞인 trace의 PolyServe·Llumnix SLO 두 구성은 각 1 run(EXP-168 r1)이다.
인스턴스↔엔진 포트는 스케줄러 로그의 `refreshInstanceMetadata … api_server_port` 줄 또는 1초 지표의 동시 처리 상관(r ≥ 0.99)으로, 요청↔엔진은 판정 로그 uuid ↔ `request_ids.jsonl`(`cmpl-<uuid>`)로 100 % 맞췄다.
llm-d도 클라이언트 요청 번호 = `"cmpl-" + Envoy x-request-id`로 정확히 맞는다(200 응답 55,699/55,699) — EXP-114 이후 기록의 "llm-d는 클라이언트와 공유하는 식별자가 없다"는 틀렸다.

### 5.1 정정 (이 문서 3절과 앞 기록에 대해)

| 이전 서술 | 고친 서술 | 근거 |
|---|---|---|
| Llumnix SLO의 예측은 받은 뒤 batch·KV가 커지는 것을 빠뜨려 낮다 | **성장은 거의 0이다.** 엔진의 실제 이후 상태로 프로파일을 다시 계산해도 판정 때 값과 1 % 안. 차이는 전부 **prefill 간섭** — 프로파일(`tpot.json`)이 decode만의 step이다 | `metrics.go:553-580`, `predict_utils.go:298-319`; Python 재현이 로그의 `predicted_tpot`와 비 1.000. decode ÷ (1 − prefill 비중) 모델이 chat 실측의 0.88~0.92 |
| PolyServe 정상상태 예측에 prefill을 넣으면 실측에 가까워진다 | **그대로 더하면 3.5~5.7배로 틀린다**(대기 chunk 하나 ≈ 520 ms를 매 step에 더함, `polyserve.go:155-170`). 맞는 것은 step에 prefill을 나눠 넣는 모델(chat 0.93~0.95) | `single_class` |
| llm-d 예측기는 KV·preemption을 모른다 | **KV 사용률과 대기 수를 입력으로 쓴다**(`prediction_server.py:576-589`). 다만 꼬리를 과소예측(TTFT 10 s 초과: 예측 0.3 % 대 실제 16.4 %) | `single_class` |
| FluidServe iteration 예측/실측 1.001 → 예측이 정확하다 | **1.001은 현재 상태의 재구성이라 맞을 수밖에 없다.** 요청을 하나 더 받으면 어떻게 되는지를 검증한 값이 아니다(받은 뒤 +4~5 ms) | `fluidserve` |
| FluidServe 첫 토큰 예측은 1.3~1.5 s 비관적 | 판정에 걸리는 구간(예측이 budget의 60~100 %)에서는 **2.2~3.4 s 비관적**(대기열 항 3.3~4.9 s). deepresearch만에서는 반대로 낙관적 | `fluidserve` |

### 5.2 정책 × trace: 덜 쓰는 원인과 더 쓰는 원인

**FluidServe v0.6**
- chat 2.5배: 거절 구간의 66~70 %에서 엔진 넷이 49.7 ms — 거절은 대부분 실제 용량 한계. 거절 구간의 23~27 %에 37~38 ms인 엔진이 하나 있었고 그것을 막은 것은 첫 토큰 판정(대기열 추정 5.13 s 대 그 엔진으로 간 요청의 실제 TTFT 중앙값 0.58 s; 거절의 약 5 %).
- swe만 (**더 씀**, +4~5 ms): 받는 순간 예측한 받은 뒤 평균 iteration의 중앙값 75.0 ms = 판정선, 그 뒤 실측 80~82 ms. ① KV 항 +2.2 ms — 투영이 h 동안 끝날 요청의 KV(16.4만 토큰)를 빼고 도착은 더하지 않아(`--fluidserve-arrival-horizons` 0) 실제보다 17만 토큰 낮다. ② prefill 항 +1.9~2.9 ms — 같은 초에 몰려 보낸 prompt(인스턴스당 초당 6.7건)가 대기 prefill에 아직 없다(`max(대기 prefill, 기대 도착)`에서 대기가 이긴 경우 0.0 %). ③ 측정된 step이 낮은 순간(77 ms, 평균 80.5)에 받는 편향. v0.5도 같은 구조(판정선 67.5, 예측 69~70, 실측 76 ms).
- deepresearch만 (**더 씀**, KV): 메모리 판정이 받는 양을 정한다(판정 기준 물리 KV 0.95, 실제 0.995). 투영이 끝날 요청의 KV(6.2만 토큰)를 빼서 5 % 여유를 먼저 쓰고, 대기 prefill(중앙값 1.2만 토큰)은 넣지 않는다. 받은 뒤 KV ≥ 0.99인 시간 80~84 %, preemption 엔진당 분당 47회.
- 섞인 trace: 거절 구간의 93~97 %에서 엔진 넷이 가장 빡빡한 budget의 3 ms 안이거나 KV ≥ 0.95 — **덜 쓰는 구간은 약 3 %**. 거절 사유 84~86 % "가장 좋은 엔진에 보내도 자기 budget 못 지킴", 14~16 % "어느 엔진이든 처리 중 요청이 넘음"; instance cap만이 막은 경우 약 1 %.
  더 쓰는 것은 swe가 가장 빡빡한 엔진에서 약 2 ms(swe만과 같은 원인, 85 엔진-분 중 21~23).

**PolyServe**
- 단일 클래스: 메모리 판정이 묶는다(swe 논리/물리 2.43 = prefix 적중 0.62에서 1/(1−0.62) ≈ 2.63과 맞다 — 같은 궤적의 호출끼리, 같은 transcript의 반복끼리 prefix를 공유). chat에서도 거절 판정의 89~90 %가 메모리이고, 그때 물리 KV 62 %인데 토큰 간격은 prefill 간섭으로 이미 약 70 ms다(정상상태 판정은 prefill을 빼서 45 ms 선을 통과).
- 섞인 trace: tier 배분이 2(chat)/1(swe)/1(deepresearch)로 **한 시간 내내 고정**된다 — tier는 가장 빈 서버가 두 번 연속 비어야 서버를 돌려주는데 chat tier는 비지 않고, swe·deepresearch는 lazy promotion으로 chat 엔진에 손님으로 먼저 가서 pool에서 서버를 가져오는 단계까지 가지 않는다.
  - **chat 엔진은 더 쓴다**: 51~68 ms(budget의 1.01~1.37배)인데 chat 요청은 `polyserve_iter_max` p90 44.3·최대 45.0 ms(= 0.9 × 50 선)로 받았다 — 예측이 실측의 0.59~0.70. 받은 chat의 54 %만 통과, 실패의 98 %가 평균 간격.
  - **swe·deepresearch 엔진은 덜 쓴다**: 물리 KV 0.29~0.44 / 0.58, 토큰 간격 budget의 0.46~0.68, preemption 0인데 deepresearch 56 %·swe 19 %를 거절. 메모리 판정의 투영이 0.98~1.00(논리/물리 1.9~2.5, 1.36~1.40).
  - **chat 엔진으로의 promotion도 막힌다**: 증가 항이 **받으려는 요청의** tier 기대 출력을 처리 중인 batch 전체에 곱한다(`forTier(request.TpotSloMs)`, `polyserve.go:121-139`). chat 약 400건을 처리 중인 엔진에 deepresearch 하나를 넣으면 985 × 401 토큰 = 용량의 0.63~0.70을 청구하므로, chat batch가 약 300을 넘으면 79~100 %의 시간 동안 메모리 판정에 걸린다. 정상상태 거절 4.8만의 92 %도 손님(swe·deepresearch)을 chat 호스트의 45 ms로 판정한 것이다.

**Llumnix SLO**
- 공통: 거절은 **100 % 게이트웨이의 5초 포기**(거절 지연 p10~p90 5.01~5.05 s, 포기 횟수 = 503 수). 예측은 decode만의 프로파일 → 실측의 0.50~0.77.
- 단일 클래스: 먼저 막는 판정은 chat TPOT 100 %, deepresearch TTFT 100 %(판정 때 예측 p50 8,955 ms 대 9,000 ms 선), swe TTFT 59~64 %·TPOT 28~31 %. **받은 swe·deepresearch가 첫 토큰에서 실패**(TTFT p50 9.6 s·22.2 s; 실패의 96.5·100 %): TTFT 필터가 게이트웨이에서 이미 기다린 시간(받은 요청의 33~43 %가 평균 2.25 s)을 budget에서 빼지 않고(`filters.go:124-157`), KV 블록을 기다리는 대기를 보지 못한다.
- 섞인 trace, 논문 구성: 엔진 넷이 모든 클래스를 비슷하게 받아 대부분 엔진에서 chat이 가장 빡빡하다. 60~84 ms(44 엔진-구간 중 26이 넘음). deepresearch가 KV를 채워(구간 4·5·10·11에 KV 1.00, 엔진당 대기 36~50, preemption 6,054) 받은 deepresearch의 30 %만 통과(실패의 99 %가 첫 토큰; 판정 때 TTFT 예측 8.63 s 대 실측 16.9 s).
  chat은 긴 deepresearch 문맥 때문에 TPOT 예측이 49 ms 이상이 되어 거절된다.
- 섞인 trace, 처리 중 요청 기준 판정(`slot75inc`): TPOT 판정의 66.5 %(81.5만/122.6만)에서 처리 중 요청의 budget이 기준이 됐다. 모든 클래스가 예측 TPOT ≤ 45 ms에서 받혀 deepresearch가 KV를 혼자 채우지 못하고(TTFT p50 1.14 s) deepresearch admitted 29.6 → 99.7, swe 43.9 → 77.4, chat 55.2 → 38.5.
  예측이 실측의 0.46~0.62라 엔진은 59~97 ms(44 엔진-구간 중 40이 넘음) — 실패가 deepresearch 첫 토큰에서 chat 평균 간격(실패의 94 %)으로 옮겨 갔다. preemption은 비슷(6,136). 각 1 run.

**llm-d**
- 단일 클래스 p90: 예측이 실측보다 높아(TPOT chat 47.7 대 44.2, swe 70.0 대 61.8 ms) 덜 쓴다. 거절 출처: chat 500(admitter) 94.5k 대 429 1.4k, swe 35.1k 대 19.0k, deepresearch 429만 85k.
  deepresearch에서 500이 없는 이유: admitter는 **KV 2 % 미만인 엔진이 하나라도 있으면 거절하지 않는다**(`latencyslo/plugin.go:133`) — r1에서 91 %의 시간 동안 그런 엔진이 있었다. 거절은 포화 평균(≥ 1인 시간 86 %, 429의 93 %가 그 시간)에서만 나온다.
  몰림의 시작: 0초부터 배치의 약 60 %가 8001, 30 %가 8003(`least`가 여유가 가장 작은 엔진에 가장 높은 점수를 주므로 먼저 짐을 받은 엔진이 더 받는다) → 35~40초에 8001 KV 0.99·대기 18~26, 포화 > 1 → 21분 유지. r2의 초반 몰림(92~98 % 8003)은 약 380초에 예측이 8003을 budget 초과로 판정하면서 끝났다(포화와는 무관).
  swe의 몰림 위반은 preemption이 아니다(최대 간격 중 preemption 1.0 %): 첫 토큰에 budget 7 s 중 3.4 s를 쓴 요청에서 8192토큰 prefill chunk를 포함한 step(약 0.57 s)들이 몰리면 그 뒤 토큰이 마감을 넘는다.
- 단일 클래스 p50: swe에서 예측이 실측보다 낮다(75 ms 초과 비율 예측 23.3 % 대 실제 42.5 %) → 더 쓴다.
- 섞인 trace: **swe·deepresearch는 chat의 50 ms에 묶여 덜 쓴다** — p50에서는 도착의 98~100 %, p90에서는 chat이 많은 구간 99~100 %·섞인 구간 73~77 %에 엔진 넷 모두 chat이 처리 중이었고, `bufferedTPOT = min(자기 SLO, podMinTPOTSLO)`로 판정한다. 받은 swe·deepresearch의 실제 간격 45~51·44~49 ms(budget의 60~67·44~49 %)인데 deepresearch 도착의 31~45 %가 500으로 거절(그때 가장 한가한 엔진의 간격 중앙값 38~43 ms).
  **chat이 빠진 엔진에서는 더 쓴다**(p90): 판정이 75·100 ms로 풀리고 `least`가 그 엔진을 KV까지 채운다 — chat이 없는 엔진으로 간 swe는 간격 62~65 ms·p90 80~89 ms·통과 0.66~0.82(chat이 있는 엔진은 0.97~0.98), KV ≥ 0.95 시간의 71~90 %가 chat 없는 엔진.
  p50은 예측이 낮아(TPOT 45.6 대 48 ms) chat을 1만~1.3만 건 더 받고 받은 chat의 35.5 %가 50 ms를 넘으며, chat이 엔진을 늘 차지해 deepresearch는 50 ms 기준에 계속 묶여 2,100~2,800건 더 거절된다.

### 5.3 요약 — 두 가지 실패와 그 아래의 공통 원인

1. **예측하는 양이 실제 step과 다르다** (더 쓰게 만든다): PolyServe·Llumnix SLO는 decode만의 step(prefill 간섭 없음)을 예측해 실측의 0.46~0.77. llm-d p50은 중앙값이라 SLO 바로 아래로 채우면 3분의 1이 넘는다. FluidServe는 prefill 간섭을 넣지만 같은 초에 몰린 도착과 h 동안의 도착을 빼서 swe에서 +4~5 ms.
2. **KV를 세는 방식이 실제와 다르다** (덜 쓰거나 더 쓰게 만든다): PolyServe는 prefix 공유를 푼 논리 토큰 + (받으려는 요청의 기대 출력 × batch 전체)라 물리 KV 30~60 %에서 멈춘다. Llumnix SLO는 KV를 보지 않아 1.00까지 간다. FluidServe는 끝날 요청의 KV를 미리 빼고 대기 prefill을 넣지 않아 0.95 기준을 넘겨 0.995까지 간다.
3. **클래스가 섞이면 판정 기준이 가장 빡빡한 클래스로 내려간다** (덜 쓰게 만든다): llm-d(`podMinTPOTSLO`)·Llumnix SLO 변경 구성·PolyServe promotion(호스트 tier로 판정)이 모두 chat의 50 ms를 swe·deepresearch에 적용한다. FluidServe는 클래스를 엔진별로 모아 chat이 없는 엔진을 75·100 ms 근처까지 쓴다 — 섞인 trace에서의 차이의 주된 원인.
4. **경로 선택이 한 엔진에 몰고, 그 한 엔진이 클러스터 전체의 판정을 정한다** (llm-d `least` + 포화 평균; PolyServe의 고정된 tier 배분).
