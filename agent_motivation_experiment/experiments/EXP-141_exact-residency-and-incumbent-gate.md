# EXP-141 — 처리 중인 요청을 관측으로 추적하면, 그리고 Llumnix\*가 처리 중인 요청의 가장 빡빡한 예산으로 판정하면 무엇이 바뀌나

세션 접두어: `exp141r`, `exp141hr`, `exp141base`, `exp141smoke`

작성 2026-10-20, **실행 전**. 코드는 바깥 저장소 브랜치 `revision_scheduler`(`bae80bb`), 설계 결정은
`project/revision.md`의 R30·R31과 "2026-10-19 — 처리 중인 요청을 관측으로 추적한다".

## 1. 무엇을 묻나

1. **(Q1) FluidServe의 장부 오차가 결과를 바꾸는가.** 논문 구성의 FluidServe는 처리 중인 요청을 추정한다(길이 분포 판정,
   그리고 장부 수가 엔진 running 수보다 많으면 먼저 보낸 요청부터 지움 — 35 req/s run에서 장부 이탈의 96%). 게이트웨이의
   완료 통지로 장부를 정확히 만들면(`--exact-residency`) 달성률·거절률·goodput이 달라지는가.
2. **(Q2) Llumnix\*의 판정 기준이 FluidServe와의 차이 중 얼마를 설명하는가.** Llumnix\*는 새 요청 자신의 TPOT 예산으로만
   판정한다. 처리 중인 요청 중 가장 빡빡한 명목 예산과 비교하게 바꾸면(`--slo-incumbent-tpot-gate`, llm-d·FluidServe와 같은
   형태) Llumnix\*가 FluidServe에 얼마나 가까워지는가.

## 2. 예상과 근거

- **Q1**: 차이가 작을 것으로 본다(모든 도착 기준 달성률 ±2점 안). 근거: 0.9 여유가 예측 오차(p99 4.5%)를 덮고 남고,
  장부 오차가 gate를 틀리게 만드는 것은 클래스가 섞인 인스턴스에서만이며, 35 req/s에서 chat 인스턴스는 거의 chat만 있다
  (10-10 재현 run의 인스턴스별 구성). 다만 정확 장부는 엔진 대기열의 요청까지 처리 중으로 세므로, 포화 구간에서 더
  보수적이 되어 거절이 늘 수 있다.
- **Q2**: Llumnix\*의 admitted 달성률이 오르고(느슨한 요청이 chat 인스턴스에 들어가 chat을 깨뜨리는 일이 줄어서) 거절률도
  오를 것이다. 모든 도착 기준 달성률은 FluidServe에 크게 못 미칠 것으로 본다 — 변형은 한 단계 예측(지금 상태 + 1)만
  보고, 같은 클래스끼리 모으는 배치·horizon 투영·pend가 없기 때문이다.

## 3. 무엇이면 틀렸다고 보는가 (판정 규칙, 실행 전에 고정)

채점 ladder95(`FS_SWE_TBT_MS=75 FS_SWE_TTFT_S=7`). 판정은 35 req/s 3회 평균으로 하고, 두 arm의 평균 차이가 두 arm의 반복
범위 중 큰 것과 **2점** 중 큰 값 이하이면 "차이 없음"이다(이 워크로드의 반복 간 편차 1.4~3.3점).

| 질문 | 틀렸다고 보는 조건 |
|---|---|
| Q1 | `fsexact`와 `fsv3capgnofrct75`의 모든 도착 기준 달성률 차이가 문턱을 넘는다(어느 방향이든). 넘으면 논문 수치가 장부 오차에 기대고 있다는 뜻이므로 R30을 논문 수준의 문제로 올린다 |
| Q2-a | `slot75inc`의 admitted 달성률이 `slot75`보다 문턱 이상 높지 않다 → 처리 중인 요청을 보호하는 판정만으로는 Llumnix\*의 SLO 위반을 줄이지 못한다 |
| Q2-b | `slot75inc`의 모든 도착 기준 달성률이 `fsv3capgnofrct75`와 "차이 없음"이 된다 → FluidServe의 이득이 판정 기준 하나로 설명된다(논문의 다른 메커니즘 주장이 약해진다) |

## 4. 정확한 설정

| 항목 | 값 |
|---|---|
| 함대 | Llama-3.1-70B × 4, TP=2, migration 끔(10-10 재구축 그대로) |
| 스케줄러 | `/home/nxclab/tools/staging/scheduler-residency2`, md5 **`33e7cc10`**(바깥 `revision_scheduler`, 장부 키 수정 커밋). 1단계의 확인 run과 `fsexact` smoke는 그 앞 빌드 `6b9b5eac`로 돌았다 — 두 빌드는 장부에 기록하는 인스턴스 키만 다르고, 그 둘은 그 키를 읽지 않는다(아래 5절 1단계 기록) |
| 게이트웨이 | `/home/nxclab/tools/staging/gateway-residency`, md5 `590242b0`, 모든 arm에서 `--notify-completion=true` |
| 드라이버 | 정적 `k8s/exp07/run_exp141_residency.sh`, 한 시간 `k8s/exp07/run_exp141_hour.sh`(EXP-108·109 드라이버의 복사본, 차이는 각 파일 머리말) |
| 워크로드 | 정적: FluidServe 계열 `mix_short_m1_t75.json`, Llumnix\* 계열 `mix_short_m1_t75fair.json`. 한 시간: `dyn60_shift_m2Am1B_b1045` 의 `_t75`/`_t75fair` |
| arm | `slot75`, `slot75inc`, `fsv3capgnofrct75`, `fsexact` |

## 5. 순서

1. **새 빌드 확인** — 플래그를 모두 끈 `fsv3capgnofrct75`, 35 req/s 8분 1회(`exp141base`). 기준: 10-10 재현 77.5 / 76.1과
   10-16 시연 77.2의 범위에서 2점 안. 벗어나면 새 빌드가 예전 동작을 내지 않는 것이므로 본 실험을 하지 않는다.
2. **smoke** — `fsexact` 4분, `slot75inc` 4분(`exp141smoke`). 볼 것: 스케줄러 지표 `scheduler_residency_requests`가 엔진의
   running+waiting 수와 같은 크기인가, `scheduler_fluidserve_retired_total{reason="completion"}`이 route 수만큼 오르는가,
   `scheduler_slo_incumbent_gate_total{binding="resident"}`가 0보다 큰가, `/complete` 실패 경고가 없는가.
3. **정적 본 실험** — 35 req/s 8분, 반복 3회가 바깥 루프, arm 넷을 안에서 번갈아(`exp141r1..3`). 약 3시간.
4. **한 시간 trace** — arm 넷 × 1회(`exp141hr1`). 1회라 판정에 쓰지 않고 경향만 본다. 약 5시간.
5. 끝나면 Sionna를 다시 켠다(`ms_dev/scripts/gpu_idle.sh start`). 다른 프로젝트의 실험이 10-21 01:50에 예약되어 있어
   그 전에 끝나야 한다.

## 6. 결과

### 1단계 (2026-10-20 11:29~12:17, 빌드 `6b9b5eac`)

- **새 빌드 확인** `261019_1938_exp141baser1_fsv3capgnofrct75_t75_rpm_2100`: 모든 도착 77.3 / admitted 99.9 / 거절 22.6% /
  goodput 13,790 tok/s, run 검사 7개 ok. 기준(10-10 재현 77.5·76.1, 10-16 시연 77.2에서 2점 안) 통과.
- **`fsexact` smoke** `261019_1953_exp141smoker1_fsexact_t75_rpm_2100`(4분): route 6,631건, 장부 이탈 이유 completion 6,425 /
  count 0 / survival 0 / age 0. 완료 통지 실패 경고 0.
- **`slot75inc` smoke** (4분): **기준이 한 번도 바뀌지 않았다** — `scheduler_slo_incumbent_gate_total{binding="own"}` 178,948,
  `resident` 0. 원인: 장부에 기록하는 키가 `LLMInstance.Id()`(`<ip>_instance<n>_<m>` 형식)였고 필터는 `GetInstanceId()`로
  조회한다. 수정해 `33e7cc10`으로 다시 빌드했고 `slot75inc` smoke를 다시 돈다(`exp141smokeb`). 이 run은 판정에 쓰지 않는다.


### 2단계 r1에서 드러난 것 — `fsexact`의 deepresearch 붕괴 (2026-10-20 14시 분석, 반복 1회 기준)

같은 세션의 `fsv3capgnofrct75`(`261019_2126_exp141r1_*`)와 `fsexact`(`261019_2141_exp141r1_*`) 비교. 채점 ladder95.

| | FluidServe (논문) | + 정확 장부 |
|---|---|---|
| 모든 도착 / admitted / 거절 | 77.3 / 99.92 / 22.7% | 76.3 / 94.76 / 19.5% |
| deepresearch admitted | 99.89 | **62.56** — 실패 665건 전부 TTFT > 10 s(토큰 사이 시간 실패 1건) |
| chat·swe admitted | 100 / 99.12 | 100 / 100 |
| deepresearch 스케줄러 대기(pend) p90 | 2.5 s | 0.5 s |
| deepresearch 보낸 뒤 첫 토큰 중앙값 / p90 | 1.7 / 3.8 s | **8.5 / 16.6 s** |
| 8002로 보낸 deepresearch / 그중 TTFT 초과 | 2,102 / 3 | 2,578 / **926(전부)** |

인스턴스 8002(메모리가 한계인 deepresearch 인스턴스), 측정 3분 이후 평균, 대시보드 기록기의 스케줄러 값과 엔진 메트릭:

| | 논문 | 정확 장부 |
|---|---|---|
| 장부의 처리 중 요청 / 엔진 running / 엔진 waiting | 162 / 165 / 3.4 | 233 / 176 / **33.8** |
| 실제 KV / 예측 outflow / 투영 KV | 837K / 80K / 773K | 927K / **212K** / 735K(실제보다 21% 낮음) |
| headroom | −350K | **+124K** |
| 엔진 KV / preemption | 96.9% / 118 | 99.6% / 367 |

해석:
1. 진행 토큰 수 j = 보낸 뒤 엔진 step 수 − prefill chunk 수. 엔진 대기열에서 기다리는 요청도 step마다 토큰을 만든 것으로
   센다. j가 부풀면 길이 분포상 "horizon 안에 끝날 확률"이 커져 outflow에 크게 들어간다. 옛 장부는 장부 수를 엔진 running
   수에 맞추며 j가 큰 요청부터 지웠고, 그 요청들이 바로 outflow를 부풀리던 요청이었다 — 두 추정 오차가 상쇄되고 있었다.
   정확 장부는 대기 중인 요청을 끝날 때까지 남기므로 outflow가 부풀어 8002가 여유 있다고 판정됐다.
2. 첫 토큰 예측은 맞았다: 8002로 보낸 deepresearch의 예상(자기 prefill + 대기열 prefill) 중앙값 14.3 s, p90 21.7 s 대 실제
   13.3 / 17.4 s. TTFT를 넘긴 926건 중 905건은 스케줄러 자신의 예상으로도 10 s를 넘었다. 논문 구성에서 첫 토큰 마감이
   feasibility에 없어서(`--fluidserve-deadline-feasible=false`, R9) 속도·KV 조건만 통과하면 보냈다.
3. 4분 smoke에서는 나타나지 않았다(엔진 대기열과 j 오차가 쌓이는 데 시간이 걸린다).

결정(사용자, 2026-10-20): 한 시간 trace는 미루고, 정적 3회를 마친 뒤 ① 대기 요청을 진행 추정에서 빼는 수정(A,
`--exact-progress`)과 ② 첫 토큰 마감 플래그(B)를 EXP-142에서 잰다.

부수 발견: 대시보드 기록기가 run이 끝난 뒤 2분 동안 다음 조건의 스케줄러 값을 그 run에 이어 적었고, run 페이지가 마지막
표본만 보고 그릴 인스턴스를 정해 FluidServe(논문) r1의 내부 값이 비어 보였다. 고쳤다(바깥 `70b37c5`). 측정에는 영향 없음.

### 3단계 정적 본 실험 결과 (2026-10-20 12:51~15:53, 35 req/s 8분 × 3회, 반복이 바깥 루프)

ladder95(`FS_SWE_TBT_MS=75 FS_SWE_TTFT_S=7`), 대시보드 채점기(analysis_scripts의 `score_run`). 평균 [최소–최대], 3회. "모든 도착"은
거절을 위반으로 세고 측정 끝에 미완료인 요청은 뺀 값. run 12개 모두 엔진 4/4, 도착률 34.9~35.0 req/s.

| arm | 모든 도착 | admitted | 거절 % | goodput tok/s | chat adm | deepresearch adm | swe adm |
|---|---|---|---|---|---|---|---|
| Llumnix SLO (`slot75`) | 21.8 [20.7–22.5] | 38.3 [36.3–40.3] | 43.0 | 6,673 | 15.9 | 77.2 [69.2–87.8] | 73.4 [64.1–81.9] |
| Llumnix SLO + 처리 중 기준 (`slot75inc`) | 23.6 [23.2–24.0] | 25.8 [25.4–26.2] | 8.6 | 6,272 | 15.4 | 99.8 | 27.7 [25.6–30.0] |
| FluidServe 논문 (`fsv3capgnofrct75`) | 77.3 [76.5–78.3] | 99.9 | 22.6 | 13,770 | 100.0 | 99.9 | 99.4 |
| FluidServe + 정확 장부 (`fsexact`) | 75.8 [74.2–76.8] | 94.9 [94.8–95.1] | 20.2 | 13,662 | 100.0 | **64.0 [62.6–64.8]** | 99.9 |

판정(3절 규칙):
- **Q1**: 모든 도착 차이 1.5점, 문턱 = max(두 arm 반복 범위 1.8·2.6, 2) = 2.6 → **"차이 없음"**(규칙상 틀리지 않음). 그러나 이 규칙이
  보는 전체 값 아래에서 deepresearch admitted가 3회 모두 62.6~64.8%로 떨어졌다(r1 분석의 붕괴가 반복마다 재현). 거절이 2.4점 줄어
  받아들인 요청이 늘고 그만큼 deepresearch가 TTFT를 넘긴 것이 서로 상쇄된 결과다. 원인과 후속은 EXP-142.
- **Q2-a**: `slot75inc`의 admitted가 `slot75`보다 12.5점 **낮다** → **예상이 틀렸다.** 처리 중인 요청의 가장 빡빡한 예산으로 판정하게
  해도 Llumnix SLO의 SLO 위반은 줄지 않았다. 거절이 43.0 → 8.6%로 줄었고(chat 거절 54 → 1.5%, deepresearch 0 → 27%, swe 0 → 41%),
  chat은 받아들여졌지만 admitted 달성률은 그대로 15%대다. 왜 chat이 받아들여진 뒤에도 실패하는지(TTFT인지 토큰 사이 시간인지)는 아직 보지 않았다.
- **Q2-b**: `slot75inc` 23.6 대 FluidServe 77.3 → "차이 없음"이 아니다. FluidServe의 이득은 판정 기준 하나로 설명되지 않는다.
