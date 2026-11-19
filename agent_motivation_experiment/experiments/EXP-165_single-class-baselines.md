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

### 논문 run의 llm-d 버전 — 찾은 것 (2026-11-19)

- **EPP: llm-d-router 커밋 `644a885639ac64ca09d6f35af3a67fe61bcc2e31`(2026-08-29 12:11 UTC, main).** 근거: EXP-114 기록(2026-09-06~08 run)이 "배포된 EPP(`llm-d-router-endpoint-picker:main`,
  커밋 `644a8856`, `--v=2`)"라고 적었다(`archive/.../EXP-114_llama8b-8instances.md:3543-3544`). manifest의 `imagePullPolicy`가 `IfNotPresent`(`deploy/llmd/router.yaml:190`)이므로
  08-31 재구축 때 받은 이미지가 10-06 컨테이너 초기화까지 그대로 쓰였다 — EXP-108·109(08-31~09-02)와 EXP-114가 같은 이미지다. Go module proxy가 그 커밋을
  `v0.10.0-rc.1.0.20260829121136-644a885639ac`로 풀어 주었다(커밋 날짜 08-29, 08-28 초기화 뒤 08-31 재구축 때의 main과 맞는다).
  같은 방법으로 확인한 초기화 전 이미지(`9eb2ff38…`)는 커밋 `cbe302bb`(08-06 12:16 커밋, 12:18 빌드)이고, 바이너리에 `-X …/version.CommitSHA=` ldflags와 `go1.26.5`,
  `sigs.k8s.io/gateway-api-inference-extension v1.5.0`이 들어 있다 — main 이미지는 커밋 몇 분 뒤에 빌드된다.
  **`644a8856`의 이미지는 레지스트리에 태그가 없다**(태그가 붙은 08~09월 이미지는 `4a0edbb`=v0.10.0-rc.1 08-11, `71f4f09`=v0.10.0 08-17, `e48584e`=v0.11.0-rc.1 09-18뿐).
  태그 없는 이미지는 digest로만 받을 수 있고 digest는 기록되지 않았다.
- **예측기(training·prediction server): 커밋 `04e44cd801d5`(0.9.0, 08-11 빌드).** 이 저장소는 커밋마다 `commit-<sha>` 태그를 붙이고, 08-31 이전 마지막 빌드가 `04e44cd`(08-11), 다음이 `0e39e36`(09-08)이다.
  digest: training `sha256:5679d69a8603fb555578c5b5e319eb856acda058ddee1355e5d3b1f662701761`, prediction `sha256:7377707db2efac33fed3740b3d541bd4c045b240195e03453a6e33e86d69eba3`(둘 다 지금 받을 수 있다).
  초기화 전 이미지(`ec7626b`, 08-03)와 다르다.
- **Envoy**: `envoyproxy/envoy:distroless-v1.33.2`(버전 태그, 초기화 전 digest `85500e28…`).
- 찾지 못한 곳: 이 기계의 run 디렉토리(10-06 초기화로 없음), Lustre 08-28 백업(논문 run 이전), `/home/nxclab/tools`의 체인 로그(digest 없음), containerd 저장소(권한 없음), 이 기계의 Claude 세션 기록(11-04 이후뿐).

### EPP를 커밋에서 다시 빌드 (2026-11-19, 사용자 결정 "직접 빌드하자")

- **절차가 upstream과 같다는 확인**: upstream `Dockerfile.epp`는 `golang:<버전>` 빌더에서 `WORKDIR /workspace`, `go mod download`,
  `CGO_ENABLED=0 go build -ldflags="-s -w -X …/version.CommitSHA=<sha> -X …/version.BuildRef=main" -o bin/epp cmd/epp/main.go`를 하고 바이너리만 distroless 이미지에 복사한다.
  `-trimpath`가 없어 소스(`/workspace`), module cache(`/go/pkg/mod`), GOROOT(`/usr/local/go`) 경로가 바이너리에 들어가므로 같은 경로의 `golang` 컨테이너(k8s Job, CPU만)에서 빌드했다
  (`/home/nxclab/tools/llmd-build/build-job.template.yaml`; pod에서 proxy.golang.org로 나가는 연결이 다른 인증서를 받아 막혀서 module cache는 호스트에서 받아 `/go/pkg/mod`에 마운트, `GOPROXY=off`).
  **같은 절차로 커밋 `cbe302bb`(`golang:1.26.5`)를 빌드한 결과가 레지스트리의 08-06 `:main` 이미지 바이너리와 바이트 단위로 같았다**(sha256 `4a168eb5…`, 72,708,258바이트).
- **`644a8856` 빌드**: 그 커밋의 `Dockerfile.epp`가 정한 `golang:1.26.7`로 같은 절차. 결과 `/home/nxclab/tools/llmd-bin/epp-644a8856`, sha256 `c735485348dac37e1ba924f3864d36f7a941aeecd6008dd024f6590a7e544bfa`,
  go1.26.7, `gateway-api-inference-extension v1.5.0`.
- **배포**: 원래 manifest는 두고 이미지를 고정한 사본을 만들었다 — `deploy/llmd/router-paper.yaml`(EPP는 08-06 이미지(같은 distroless/static:nonroot 바탕)를 그릇으로 쓰고
  `command: /llmdbin/epp-644a8856`을 hostPath로 마운트, Envoy는 digest 고정), `deploy/llmd/predictor-paper.yaml`(`04e44cd` digest 둘). namespace `llmd`, CRD 둘, `pathb.yaml`, `epp-config-slo.yaml`은 원래 파일 그대로.
  엔진이 내려가 있어 endpoints ConfigMap은 자리표시자로 만들었다(경로 B는 읽지 않고, 드라이버가 run마다 `llmd_endpoints.py`로 다시 쓴다).
- **기동 확인**(엔진 없이): EPP가 설정을 읽고 우리 설정의 plugin을 모두 만들었다(`streamingMode: true`, `slo-headroom-tier-filter`, `latency-slo-admitter`), InferencePool·InferenceObjective를 reconcile했다.
  **EPP 지표 `llm_d_epp_info{build_ref="main",commit="644a885639ac64ca09d6f35af3a67fe61bcc2e31"}`** — EXP-114가 "커밋 644a8856"을 읽은 것과 같은 지표다.
  training server는 표본 0개라 학습을 건너뛰고, prediction server의 gated model 요청에 404를 낸다 — 첫 run에서 표본이 쌓인 뒤 사라지는지 확인한다.
- **드라이버** `k8s/exp07/run_exp165_llmd.sh`: 논문 드라이버 `run_exp109_llmd.sh`의 `full`(한 시간 trace)과 같고 바뀐 것은 변형별 trace·fair 설정 선택, chat 2.5배에서 부하 생성 36개,
  엔진 재시작 뒤 `llmd_endpoints.py`(InferencePool 포트 동기화, `run_exp139_llmd.sh`에서), 그리고 preflight에서 이미지·바이너리 sha256, 예측기 재시작 뒤 EPP가 보고하는 커밋을 검사해 다르면 ABORT.
  run마다 엔진(`neutral-0`)과 예측기 셋·EPP를 재시작하고 training server가 표본 0개에서 시작하는지 확인한다(논문 run과 같다; 측정 한 시간 동안 예측기가 학습한다).
- **실행**: 체인 `/home/nxclab/scratch/exp165l/run.sh`, 로그 `/home/nxclab/tools/exp165l_chain.log`. r1 = onlychat25, onlyswe, onlydr; r2 같은 순서. 6 run, 약 7.5시간.
  판정은 위 판정 절과 같다(PolyServe·Llumnix SLO와는 세션이 하루 다르므로 세션 간 문턱을 쓴다).

### ⚠ 첫 시도 무효 — 예측기가 학습하지 못했다 (2026-11-19 12:51~13:13 KST)

- **무효 run**: `results/261118_1951_exp165lr1_llmdslot75_onlychat25`(약 22분 측정 뒤 체인과 runner를 멈췄다; 지우지 않는다, 쓰지 않는다). 다시 걸 때 세션 이름은 `exp165m`.
- **무엇이 틀렸나**: 예측기를 `04e44cd`(0.9.0, 08-11)로 고정했었다. 그 `training_server.py`는 모듈 수준 함수 `_drop_timestamp`(60행, `@staticmethod`가 모듈 수준에 붙어 있다)를
  `self._drop_timestamp(...)`로 부른다(714·859·968·1046행). 그래서 `train()`이 매번 `AttributeError`로 실패하고(`Initiating training with 52855 samples … Critical error in train(): 'LatencyPredictor' object has no attribute '_drop_timestamp'`)
  모델이 만들어지지 않는다. EPP는 그때 모든 요청에 **TTFT·TPOT 예측 10 ms 고정값**을 썼다(예측 합/개수 431.69 s / 43,169). 그러면 latency-slo-admitter는 모든 엔진이 SLO 안이라고 보아 거절하지 않고,
  그 run의 429(5,413건, 약 25 %)는 EPP의 legacy 포화 기반 admission이 만든 것이다 — llm-d의 SLO 인식 경로가 동작하지 않은 run이다. 드라이버의 검사(이미지·바이너리·커밋·표본 0개 시작)는 모두 통과했다.
- **논문의 예측기는 `04e44cd`일 수 없다**: EXP-114(09-06~08, 같은 이미지가 쓰인 기간)가 EPP의 예측·실측 히스토그램으로 TTFT 예측/실측 1.18, TPOT 1.10을 쟀다 — 학습된 모델의 값이다.
- **`ec7626b`로 바꾼 근거**: 레지스트리의 태그 모양에서 release 빌드는 `<버전>`과 `commit-<sha>`를 붙이고 `main`은 옮기지 않는다(1.0.0 = `commit-c810ebc` = `latest`, 10-16; 지금 `main`은 11-18의 별도 main 빌드).
  `commit-04e44cd`는 `0.9.0`·`0.9`·`0`과 digest가 같은 release 빌드이고, main 빌드는 `ec7626b`(08-03) 다음이 `0e39e36`(09-08)이다. 따라서 08-31의 `:main`은 `ec7626b`였을 가능성이 가장 높다
  (초기화 전 pod가 쓰던 digest와 같다: training `490d6bb9…`, prediction `8322e604…`). `ec7626b`에는 문제의 코드가 없다. `deploy/llmd/predictor-paper.yaml`을 이 digest로 바꿨다.
- **기능 확인(측정 run 아님)**: 엔진이 떠 있는 동안 llm-d router로 짧은 요청 520건(헤더 5000/50, sheddable)을 보냈다. training server가 `TTFT model trained on 358 samples … Coverage = 95.35 %`,
  `TPOT model trained on 357 samples`를 찍었고, 그 뒤 120건의 EPP 예측 평균은 TPOT 17.6 ms·TTFT 64.0 ms(10 ms 고정이 아니다; EXP-114의 예측 TPOT 평균 18.6 ms와 같은 크기). 드라이버가 run마다 예측기를 0개에서 다시 시작하므로 이 요청은 측정에 남지 않는다.
- **드라이버에 넣은 검사**: run 시작 10분 뒤 training server 로그에 `model trained on`이 있고 `Critical error in train`이 없어야 하고, run 시작 뒤 EPP 예측 TTFT 평균이 10 ms 고정값이 아니어야 한다 — 아니면 runner를 지우고 ABORT.
  알려진 답 셋(지금 상태 = 통과, 10 ms 고정 = ABORT, 로그에 crash 줄 = ABORT)으로 확인했다. 처음 쓴 판정은 `echo "$log" | grep -q`였는데 드라이버가 `pipefail`이라 `grep -q`가 먼저 끝나면 `echo`가 SIGPIPE로 실패해
  "학습됨" 검사는 늘 실패하고 "crash" 검사는 절대 발동하지 않았다 — 알려진 답 시험에서 잡고 here-string으로 바꿨다.
