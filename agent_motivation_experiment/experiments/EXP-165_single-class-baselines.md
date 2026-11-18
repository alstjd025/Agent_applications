# EXP-165: 클래스 하나만 있는 한 시간 trace — 기준선 세 정책 (Llumnix SLO, PolyServe, llm-d)

**날짜**: 2026-11-18 (실행 전 작성)
**질문**: 클래스가 하나뿐일 때(chat 2.5배 / swe만 / deepresearch만) 기준선 정책들은 얼마나 받고, 받은 요청의 마감을 얼마나 지키는가.
EXP-164에서 FluidServe v0.6은 같은 세 trace에서 admitted 99.94 / 97.68 / 97.35였다. 기준선이 같은 trace에서 어디서 실패하는지를 보아,
v0.6의 남은 두 문제(swe만에서 받는 순간의 토큰당 시간 추정이 실제보다 낮다, deepresearch만에서 KV 판정이 너무 많이 받는다)가
FluidServe에만 있는 문제인지, 단일 클래스 부하 자체가 만드는 문제인지를 가린다. (2026-11-18 사용자 요청: "llumnix-slo(논문 버전에서 변경한거),
polyserve(논문 버전), llm-d(논문 버전)를 각 워크로드만으로 실험")

## 구성

### 정책 (논문 구성 확인 결과)

| 정책 | arm | 설정 | 근거 |
|---|---|---|---|
| **Llumnix SLO** (리비전에서 바꾼 판정) | `slot75inc` | `--scheduling-policy slo` + `SLO_INCUMBENT_TPOT_GATE=1`: 새 요청을 자기 TPOT budget과, 그 인스턴스에서 처리 중인 요청 중 가장 빡빡한 TPOT budget 중 작은 값의 0.9배로 판정. TTFT·TPOT 필터 0.9, 게이트웨이 1,000 ms마다 재시도·5,000 ms 뒤 포기(게이트웨이 기본값) | `run_exp165_hour.sh` set_arm, `project/revision.md` R31, `pkg/scheduler/policy/filters.go:136-150` |
| **PolyServe** (논문 구성) | `polyservept75` | `PS_ADMISSION_BINDS=true PS_STEADY_NO_PREFILL=true PS_PROMOTION=true PS_PARTITION=elastic PS_PREFER_LOADED=true PS_KV_ADMISSION=true`, 게이트웨이 기본값 | 논문 run의 드라이버 `run_exp108_t75.sh:329-332`, `run_exp109_hour.sh:523-526`와 같다. PolyServe 결정 코드는 2026-09-23(`2e752f2`, 지표 공개만) 이후 바뀌지 않았다 |
| **llm-d** (논문 구성) | `llmdslot75` | upstream llm-d의 `predicted-latency-slo` 구성(Envoy + EPP, InferencePool 경로, `streamingMode: true`, `latency-slo-admitter`, InferenceObjective `sheddable` 우선순위 −1)이 같은 엔진 4대로 route. 클라이언트가 `x-llm-d-slo-ttft-ms`·`x-llm-d-slo-tpot-ms` 헤더를 보낸다 | `archive/llumnix_reproduce/ms_dev/notes/llmd-baseline.md`, `deploy/llmd/`, 논문 run 드라이버 `run_exp109_llmd.sh full`. **스택을 다시 세운 뒤 추가한다(아래 llm-d 절)** |

**요청이 한 클래스만 들어올 때 두 정책의 동작 (코드로 확인)**:
- PolyServe의 elastic 파티션은 클래스 비율 설정을 읽지 않는다. 모든 인스턴스가 남는 pool에서 시작하고, 어떤 클래스의 인스턴스가 모두 요청을 거절하면 그 클래스가 pool에서
  인스턴스를 하나 더 가져온다(`polyserve.go:746-783`, `polyserve_pool.go:79-85,147-170`). 그래서 한 클래스가 인스턴스 4대를 모두 쓸 수 있고, 없는 클래스 몫으로 인스턴스를 비워 두지 않는다.
- Llumnix SLO의 바뀐 판정은 처리 중 요청의 budget이 새 요청의 budget보다 작을 때만 결정을 바꾼다(`filters.go:142`). 클래스가 하나면 모든 요청의 budget이 같으므로
  **`slot75inc`의 결정은 논문의 Llumnix SLO(`slot75`)와 같다.** 이 실험의 Llumnix SLO 결과는 두 판정 모두의 결과로 읽는다. 확인: 스케줄러 지표
  `scheduler_slo_incumbent_gate_total{binding="resident"}`가 0이어야 한다.

### 워크로드 설정 — 기준선은 swe `tbt_ms` 75인 "fair" 설정

PolyServe·Llumnix SLO·llm-d는 설정 파일의 `slo.<class>.tbt_ms`를 토큰당 목표로 그대로 읽는다. FluidServe의 설정은 swe에 25를 적고(클래스 구분값) 실제 75 ms를
`--fluidserve-class-budgets 25:decode:75`로 받으므로, 같은 파일을 기준선에 주면 swe를 budget의 1/3로 판정한다. 논문 trace에서 기준선에 `..._t75fair.json`을 준 것과 같은 이유로
세 변형에 fair 설정을 새로 만들었다(원본과의 차이는 swe `tbt_ms` 25 → 75 하나, 스크립트로 대조):

| 변형 | trace | 기준선 설정 | 부하 생성 프로세스 |
|---|---|---|---|
| `onlychat25` | `dyn60_onlychat_b25112.csv` (EXP-164와 같다, 초당 25~112.5건, 249,234건) | `mix_dyn60_onlychat_b25112_t75fair.json` | 36 |
| `onlyswe` | `dyn60_onlyswe_b1045.csv` | `mix_dyn60_onlyswe_b1045_t75rfair.json` (swe 반복 번호 켬) | 12 |
| `onlydr` | `dyn60_onlydr_b1045.csv` | `mix_dyn60_onlydr_b1045_t75fair.json` | 12 |

chat만과 deepresearch만에서는 swe 요청이 없으므로 요청의 budget은 원래 설정과 같다.

**PolyServe의 tier 표는 이 설정 파일에서 만들어지지 않는다 — 논문의 한 시간 trace run과 같다 (2026-11-18 첫 run에서 발견).**
PolyServe가 tier마다 쓰는 기대 출력 길이 표(`--polyserve-tier-decode-tokens`)는 `PS_MIX_CONFIG`가 가리키는 설정 파일의 swe·chat·deepresearch `tbt_ms`를 key로 만든다
(`ms_dev/scripts/set_scheduler_profiling.py` `tier_by_class`). 한 시간 trace 드라이버는 run_cell에서 `PS_MIX_CONFIG`를 export한 뒤 `set_arm`에서 다시 unset한다
(`run_exp109_hour.sh:190-191`, 논문 run의 드라이버; `run_exp165_hour.sh:381-382`도 같다). 그래서 기본 파일 `mix_short_m1_balanced.json`(swe `tbt_ms` 25)에서 표가 만들어져
`25:494,50:428,100:985`가 된다(EXP-165 첫 run의 기동 줄). swe 요청은 토큰당 75 ms를 달고 오므로 표에서 찾지 못하고, 기본값 463토큰(`PolyserveDecodeTokens`)을 쓴다 —
swe의 평균 출력 494토큰보다 6 % 짧게, 앞으로의 KV 증가를 작게 투영한다. 정적 부하 드라이버(`run_exp108_t75.sh`)는 unset하지 않아 `50:428,75:494,100:985`였다
(`results/exp07_meta/scheduler_deploy_rpro1010pr1_polyservept75_t75fair.yaml`, `evidence.md` §3.2 줄도 이 값).
- 영향: chat(50)·deepresearch(100) 항목은 두 표가 같으므로 `onlychat25`·`onlydr`에는 영향이 없다. `onlyswe`의 PolyServe만 영향을 받는다.
- 논문 PolyServe 한 시간 run(EXP-109, 2026-09)도 같은 표로 돌았는지는 그 run의 `scheduler_deploy_*.yaml`(맥 SSD)로 확인해야 한다 — 드라이버의 순서로는 같았다.
- **고쳤다 (2026-11-18 16:09 KST, 사용자 지시 "polyserve는 SLO 바르게 잡도록 고쳐 제대로. kv 투영도 제대로 하고")**: `run_exp165_hour.sh`의 `set_arm`이 `PS_MIX_CONFIG`를
  지우지 않게 하고, PolyServe arm은 스케줄러가 받은 tier 표가 그 run의 설정 파일에서 만든 표(`50:428,75:494,100:985`)와 다르면 ABORT하게 했다(현재 배포된 틀린 표로 검사가
  불일치를 내는 것을 먼저 확인). 체인 실행 중이라 새 파일을 쓰고 `mv`로 바꿔 넣었다 — 드라이버는 run마다 새로 실행되므로 다음 run부터 적용된다. 고치기 전 파일:
  `/home/nxclab/scratch/exp165/run_exp165_hour.sh.before-tierfix`.
- 고치기 전에 돈 PolyServe run은 r1 `onlychat25` 하나다(표 `25:494,50:428,100:985`). chat 요청은 50 항목(428)만 쓰고 두 표의 50 항목이 같으므로 결정은 고친 표와 같다.
  swe만 trace의 PolyServe run 둘은 고친 표로 돈다.
- 고치지 않은 것: §4.5 KV 투영의 구조(처리 중 요청마다 이미 만든 토큰과 상관없이 tier의 기대 출력 전체를 더하고, prefix 공유를 풀어 센 토큰 수를 물리 KV 용량과 비교한다).
  EXP-127이 이것 때문에 관측 KV 60 % 근처에서 메모리 거절이 난다는 것을 쟀고, 2026-09-11 사용자 결정으로 논문 충실성 때문에 고치지 않았다.

### 실행

| 항목 | 값 |
|---|---|
| 바이너리 | 스케줄러 `7dc2355c`, 게이트웨이 `05745270` (EXP-164와 같다) |
| 드라이버 | `k8s/exp07/run_exp165_hour.sh` — `run_exp164_hour.sh` 복사. EXP-164 드라이버는 기준선의 fair 설정을 shift 변형에서만 골랐다. 그대로였으면 swe만에서 기준선이 swe를 25 ms로 판정했다. 지금은 변형마다 fair 설정을 고르고, fair 설정이 없는 변형이면 ABORT |
| 체인 | `/home/nxclab/scratch/exp165/run.sh`, 로그 `/home/nxclab/tools/exp165_chain.log`. run마다 `run_config.json`의 클래스 계획, swe TTFT 7000, swe `tbt_ms` 75, 반복 번호, `load_procs`를 확인 |
| 순서 | 반복이 바깥. r1 = onlychat25(PolyServe → Llumnix SLO), onlyswe(같음), onlydr(같음); r2 = 같은 변형 순서, arm 순서 반대. 12 run, 약 14.5시간 |
| 채점 | ladder95, `FS_SWE_TBT_MS=75 FS_SWE_TTFT_S=7`. 분모는 `agent != "job_summary"`인 요청(task_id 기준) |

## 판정 (실행 전에 정함; 변형마다 따로)

1. 지표: 모든 도착, admitted, 거절률(클래스가 하나라 클래스별 지표는 이것과 같다), token goodput.
2. **기준선끼리**(같은 세션): "평균 차이 > max(문턱, 두 조건 각각의 반복 범위)"이면 "다르다". 문턱은 모든 도착·거절률 4점, admitted 1점.
3. **FluidServe v0.6과의 대조**: v0.6 값은 EXP-164(전날 세션)의 것이다. 세션을 넘는 비교이므로 문턱에 세션 간 편차(최대 4.6점)를 더해
   모든 도착·거절률은 max(4.6, 반복 범위), admitted는 max(1, 반복 범위)를 넘을 때만 "다르다"고 적고, 그보다 작으면 "같은 세션에서 다시 재야 판정할 수 있다"고 적는다.
   preemption 횟수는 세션을 넘어 비교하지 않는다.
4. 같이 적는 것: 받은 요청의 마감 위반 종류(첫 토큰 / 평균 토큰 간격 / 몰림), TTFT p50/p90/p99, 평균 토큰 간격 분포, iteration 실측, preemption, prefix 적중,
   거절 사유(PolyServe: 메모리 거절과 토큰당 시간 거절의 비율, 게이트웨이 5초 포기), 클라이언트 상태(분마다 요청 수 ÷ trace, 거절이 아닌 오류).
5. 해석의 초점:
   (가) swe만에서 기준선의 받은 요청도 평균 토큰 간격으로 실패하는가 — 그렇다면 "받는 순간의 추정이 받은 뒤의 실제보다 낮다"는 FluidServe만의 문제가 아니라 swe 부하의 성질이다.
   (나) deepresearch만에서 기준선도 preemption이 많은가 — PolyServe는 KV admission(`PS_KV_ADMISSION`)이 있고 Llumnix SLO는 없다.
   (다) PolyServe가 대기열에 넣은 swe(첫 토큰 budget 7 s)·deepresearch(10 s) 요청은 게이트웨이가 5 s에 포기하므로 자기 budget 전에 거절된다(코드에서 확인, 측정 아님) — 거절 지연 분포로 확인.

**예상**: Llumnix SLO는 admission이 TTFT·TPOT 예측 필터 하나라 세 trace 모두에서 많이 받고 admitted가 v0.6보다 낮다(deepresearch만에서 preemption이 v0.6과 비슷하거나 많다).
PolyServe는 메모리 거절이 많아 거절률이 높고 admitted는 높다.

## 결과 (2026-11-18 14:12 ~ 11-19 04:44 KST, 12 run, ladder95)

체인 `=== EXP165 CHAIN DONE 2026-11-19 04:44:42`(`bin/` 복원 `3251836a`/`590242b0`, 게이트웨이 replica 0, Sionna 복귀). ABORT·WARN 0. 12 run 모두 체인의 설정 확인 통과
(클래스 계획, swe TTFT 7000·`tbt_ms` 75, 반복 번호, `load_procs`). PolyServe는 r1 `onlychat25` 하나만 고치기 전 표(`25:494,50:428,100:985`, chat에는 같은 결정)였고
나머지 다섯은 `50:428,75:494,100:985`를 드라이버가 확인했다. 모든 run에서 분마다 요청 수 ÷ trace 0.989~1.014, 전체 도착 수가 trace와 같고 거절이 아닌 오류 0.
Llumnix SLO의 처리 중 요청 기준 판정은 여섯 run 모두 `binding="own"`만 올라갔다(265만~486만 회, `resident` 0) — 논문의 Llumnix SLO와 같은 결정이다.
집계: `analysis_scripts/exp163/observe.py`(run별 `/home/nxclab/scratch/exp165/obs_*.txt`; 다른 정책의 run에서 FluidServe 전용 부분을 건너뛰도록 고쳤다), 요약 `/home/nxclab/scratch/exp165/summary.py`.

| 변형 / 정책 | 모든 도착 r1·r2 | admitted r1·r2 | 거절률 r1·r2 | 받은 요청의 마감 위반 (첫 토큰/평균 간격/몰림) r1 | TTFT p50/p99 | 평균 토큰 간격 p50/p90 | 엔진 KV 사용률 최대 | preemption r1·r2 |
|---|---|---|---|---|---|---|---|---|
| chat 2.5배 / PolyServe | 20.43 · 19.44 | 29.65 · 28.52 | 31.0 · 31.7 | 119,538 (3,492/115,557/489) | 0.64/5.61 s | 71.5/76.2 ms | 0.69 · 0.71 | 0 · 0 |
| chat 2.5배 / Llumnix SLO | 21.55 · 21.41 | 31.46 · 31.22 | 31.4 · 31.3 | 115,817 (2,429/113,313/75) | 0.58/5.53 s | 81.7/87.6 ms | 0.82 · 0.82 | 0 · 0 |
| swe만 / PolyServe | 41.98 · 42.12 | 99.72 · 99.83 | 57.9 · 57.8 | 116 (16/1/99) | 1.16/6.28 s | 52.3/55.7 ms | 0.44 · 0.44 | 0 · 0 |
| swe만 / Llumnix SLO | 2.66 · 2.64 | 4.85 · 4.82 | 44.8 · 44.8 | 51,263 (49,453/1,807/3) | 9.56/15.17 s | 116.8/122.3 ms | 1.00 · 1.00 | 9,302 · 9,109 |
| deepresearch만 / PolyServe | 31.72 · 31.60 | 100.00 · 100.00 | 68.2 · 68.3 | 0 | 1.10/6.06 s | 48.9/51.2 ms | 0.71 · 0.69 | 0 · 0 |
| deepresearch만 / Llumnix SLO | 0.47 · 0.45 | 0.97 · 0.93 | 51.3 · 51.3 | 46,894 (46,893/0/1) | 22.20/36.51 s | 77.6/82.0 ms | 1.00 · 1.00 | 20,623 · 20,616 |

모든 run이 반복 간 0.0~1.1점으로 안정적이다(가장 큰 것은 PolyServe chat 2.5배의 admitted 1.13점).

**PolyServe의 거절 이유** (스케줄러 `scheduler_polyserve_refused_total`의 run 동안 증가분; 후보 인스턴스마다·게이트웨이 재시도마다 세므로 요청 수가 아니다):

| 변형 | memory | steady_state | second_token | first_token |
|---|---|---|---|---|
| chat 2.5배 r1 · r2 | 1,876,081 · 1,934,464 | 234,504 · 215,483 | 0 · 1 | 31 · 28 |
| swe만 r1 · r2 | 1,501,364 · 1,501,933 | 0 · 0 | 58 · 71 | 0 · 4 |
| deepresearch만 r1 · r2 | 1,736,830 · 1,743,912 | 0 · 0 | 22 · 22 | 2 · 0 |

거절하는 순간의 투영 KV는 인스턴스 용량(584,928토큰)을 막 넘는 값(swe만 r1 약 588,000)인데, 엔진이 실제로 쓴 KV는 최대 44 %(swe만)·69~71 %(deepresearch만·chat)였다.

### 판정 (지표마다; 같은 세션 PolyServe 대 Llumnix SLO는 문턱 4/1점, v0.6과는 세션이 달라 max(4.6, 반복 범위)/max(1, 반복 범위))

| 변형 | 비교 | 모든 도착 | admitted | 거절률 |
|---|---|---|---|---|
| chat 2.5배 | Llumnix SLO 대 PolyServe | 19.94 → 21.48 (+1.54) 차이 없음 | 29.09 → 31.34 (+2.25, 기준 1.13) **Llumnix SLO가 좋음** | 31.33 → 31.35 차이 없음 |
| | PolyServe 대 v0.6 | 68.16 → 19.94 (−48.2) **나쁨** | 99.94 → 29.09 (−70.9) **나쁨** | 31.75 → 31.33 차이 없음 |
| | Llumnix SLO 대 v0.6 | 68.16 → 21.48 (−46.7) **나쁨** | 99.94 → 31.34 (−68.6) **나쁨** | 31.75 → 31.35 차이 없음 |
| swe만 | Llumnix SLO 대 PolyServe | 42.05 → 2.65 (−39.4) **나쁨** | 99.78 → 4.83 (−94.9) **나쁨** | 57.81 → 44.76 (−13.1) **좋음** |
| | PolyServe 대 v0.6 | 54.01 → 42.05 (−12.0) **나쁨** | 97.68 → 99.78 (+2.09) **좋음** | 44.60 → 57.81 (+13.2) **나쁨** |
| | Llumnix SLO 대 v0.6 | 54.01 → 2.65 (−51.4) **나쁨** | 97.68 → 4.83 (−92.9) **나쁨** | 44.60 → 44.76 차이 없음 |
| deepresearch만 | Llumnix SLO 대 PolyServe | 31.66 → 0.46 (−31.2) **나쁨** | 100.00 → 0.95 (−99.1) **나쁨** | 68.26 → 51.31 (−16.9) **좋음** |
| | PolyServe 대 v0.6 | 38.09 → 31.66 (−6.4) **나쁨** | 97.35 → 100.00 (+2.65) **좋음** | 60.60 → 68.26 (+7.7) **나쁨** |
| | Llumnix SLO 대 v0.6 | 38.09 → 0.46 (−37.6) **나쁨** | 97.35 → 0.95 (−96.4) **나쁨** | 60.60 → 51.31 (−9.3) **좋음** |

### 해석

- **Llumnix SLO는 세 trace 모두에서 받은 요청을 지키지 못한다**(admitted 31 / 4.8 / 0.95). 실패하는 방식이 trace마다 다르다:
  chat 2.5배에서는 받은 요청의 약 67 %가 평균 토큰 간격으로 실패한다(실제 82 ms, budget 50 ms; TPOT 필터가 새 요청 하나의 decode 시간을 0.9 × 50 ms와 비교하는데,
  계속 섞여 들어오는 prefill이 decode iteration을 늘리는 것을 그 예측이 반영하지 못하는 것으로 보인다 — 코드에서의 추론이고 측정하지 않았다).
  swe만·deepresearch만에서는 KV를 보는 admission이 없어 엔진 넷 모두 KV 100 %에 닿고 preemption이 9천·2만 회 일어나며, preemption된 요청이 prefill을 다시 해서
  받은 요청 거의 전부가 첫 토큰 budget(7 s·10 s)을 넘긴다(TTFT 중앙값 9.6 s·22.2 s).
- **PolyServe는 KV 투영이 실제보다 커서 엔진 KV를 44~71 %까지만 채우고 거절한다**(거절 판정의 89~100 %가 memory). EXP-127이 8 인스턴스에서 잰 것(관측 60 %에서 거절)과 같은 원인 —
  처리 중 요청마다 이미 만든 토큰과 상관없이 tier의 기대 출력 전체를 다시 더하고, prefix 공유를 풀어 센 토큰을 물리 용량과 비교한다. swe는 prompt가 길고 앞부분을 공유해 이 차이가 가장 크다(44 %).
  그래서 swe만·deepresearch만에서는 받은 요청을 거의 다 지키지만(99.8 / 100.0) v0.6보다 6~12점 덜 받는다.
  chat 2.5배에서는 KV에 여유가 있어 많이 받는데, 정상상태 iteration 추정(§4.5대로 prefill을 뺀다, `--polyserve-steady-state-ignores-prefill`)이 prefill이 섞인 실제 iteration을 반영하지 못해
  받은 chat의 약 68 %가 평균 토큰 간격으로 실패한다(실제 72 ms).
- **FluidServe v0.6과의 대조(세션이 다르다)**: chat 2.5배에서는 세 정책의 거절률이 같고(31~32 %) v0.6만 받은 요청을 지킨다(admitted 99.9 대 29~31). swe만·deepresearch만에서는
  v0.6이 더 받고(모든 도착 +6.4~+12.0 대 PolyServe) PolyServe가 받은 것을 더 잘 지킨다(admitted +2.1~+2.7). v0.6의 남은 두 문제(swe만의 평균 토큰 간격 초과 2.3 %, deepresearch만의
  preemption 약 1.2만 회와 첫 토큰 실패 약 1,000건)는 PolyServe에서는 나타나지 않는데, 엔진 KV를 절반 남짓만 쓰기 때문이다. Llumnix SLO는 같은 단일 클래스 trace에서 같은 두 문제를
  훨씬 크게 보인다 — 받는 순간의 토큰당 시간 추정이 실제보다 낮은 문제와 KV를 넘치게 받는 문제는 단일 클래스 부하에서 KV를 충분히 쓰는 정책 모두에 나타난다.
- 예상 대조: "Llumnix SLO는 많이 받고 admitted가 v0.6보다 낮다" — 거절률은 v0.6과 같거나 낮았고 admitted는 예상보다 훨씬 낮았다(1~31). "PolyServe는 메모리 거절이 많아 거절률이 높고
  admitted가 높다" — swe만·deepresearch만에서 맞았고, chat 2.5배에서는 거절률이 v0.6과 같고 admitted가 29로 틀렸다.

## llm-d (스택 재구축 뒤 추가)

클러스터를 2026-10-10에 다시 세울 때 llm-d 스택은 세우지 않았다(namespace `llmd`, InferencePool·InferenceObjective CRD 없음). manifest(`deploy/llmd/`)와 드라이버는 남아 있고,
엔진은 Llumnix와 같은 `neutral-0`의 4대를 같이 쓴다(llm-d는 GPU를 쓰지 않는다). 이미지가 노드에 없고 manifest는 바뀌는 태그(`:main`)를 가리킨다.

- 2026-08-28 컨테이너 초기화 전 백업(`/NHNHOME/storage/home_backup_2026-08-28/cluster_state/all.yaml`)의 이미지: EPP `9eb2ff38…`(2026-08-06 빌드),
  training·prediction server `490d6bb9…`·`8322e604…`(2026-08-03 빌드, 커밋 `ec7626be00b4`), Envoy `distroless-v1.33.2`(`85500e28…`). 모두 지금도 ghcr에서 받을 수 있다(2026-11-18 확인).
- **논문 run(EXP-108·109, 2026-08-31~09-02)은 초기화 뒤 `:main`을 다시 받아 돌았고, 그 digest는 기록되지 않았다.** 2026-08-06 이후 `:main`은 바뀌었다(EPP v0.10.0이 08-17 빌드).
  지금의 `:main`은 2026-11-17 빌드다.
- 어느 이미지로 다시 세울지는 사용자가 정한다. 정해지면 이 절에 이미지 digest, 재구축 절차, 첫 요청 확인 결과를 적고 같은 세 변형 × 2회를 돌린다.
