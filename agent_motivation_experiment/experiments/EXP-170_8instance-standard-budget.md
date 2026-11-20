# EXP-170: 인스턴스 8대(Llama-3.1-8B, TP=1) — 표준 budget, FluidServe v0.6 대 PolyServe

**날짜**: 2026-11-20 (실행 전 작성)
**질문**: 논문의 세 번째 함대(Llama-3.1-8B × 8)에서 FluidServe v0.6이 **표준 budget**으로도 PolyServe보다 나은가. 논문은 이 함대를 **budget을 절반으로 줄여** 실었고(EXP-126: FluidServe offered 73.5 대 PolyServe 44.8),
같은 함대·같은 trace에서 표준 budget으로는 **PolyServe가 앞섰다**(EXP-125, 1회, v0.4 시절: PolyServe offered 76.0 대 FluidServe 68.2; 옛 채점 창). `evidence.md` §4.8·§6 6번이 리비전에서 가장 공격받기 쉬운 곳으로 꼽은 부분이다.
(2026-11-20 사용자 결정: "둘 다 해보자. 일단 표준으로 우리거랑 polyserve만" — 표준 budget을 먼저, 절반 budget은 뒤에.)

## 구성 (논문 함대와 같게; EXP-114·126, STATUS 2026-09-05 절)

| 항목 | 값 | 바꾸는 방법 |
|---|---|---|
| 모델·함대 | Llama-3.1-8B-Instruct(snapshot `0e9e39f2…`), **8대 × TP=1**, max-model-len 40960 | `ms_dev/scripts/switch_model.py --to llama31-8b-b200-tp1` (LWS·게이트웨이 tokenizer·`configmap/llumnix-model`) |
| 엔진 인자 | **`--max-num-seqs 4096`, `--api-server-count 4`**(+ 스크립트가 넣는 포트 export와 llumlet client-index 수정), liveness 3 × 30 s 그대로 | `ms_dev/scripts/set_api_server_count.py --count 4 --max-num-seqs 4096` |
| 게이트웨이 | **`-v 0`, CPU limit 24, `GOMAXPROCS` 24**(지금 70B는 `-v 3`, 8, 16). `-v 3`에서는 게이트웨이 로그가 토큰을 101개씩 뭉쳐 8B 함대의 정책 비교를 무효로 만들었다(STATUS 2026-09-05) | 체인이 deploy를 고친다 |
| 프로파일 | `deploy/profiling/llama31-8b-b200-tp1`(`FS_PROFILE_DIR`; 드라이버가 ConfigMap의 `PROFILE_DIR`과 다르면 ABORT). 구조가 70B 프로파일과 같다(메타 주석 셋만 다름). 클래스 평균 출력 swe 464·chat 403·deepresearch 1,094 |
| trace | `dyn60_shift_m2Am1B_b1045_x620.csv` — 논문의 6.20배 한 시간, **615,228건**, 분 단위 61.5~279.3 req/s, 61분. canonical CSV가 없어서 `traces/upscale_trace_keepdur.py --factor 6.2`(seed 42 기본값)로 다시 만들었다(도착 수가 논문 기록과 같다; md5 `d2cf3de7…`) |
| 워크로드 설정 | FluidServe `mix_dyn60_shift_m2Am1B_b1045_x620_t75.json`(swe tier key 25), PolyServe `..._x620_t75fair.json`(swe 75 ms). 70B 섞인 trace 설정과 trace만 다르다 | |
| 부하 생성 | 36 프로세스(EXP-126과 같다) | |
| arm | `fsg10dl`(v0.6, 스케줄러 `7dc2355c`, 게이트웨이 `05745270`), `polyservept75`(논문 구성; tier 표는 8B 프로파일의 클래스 평균에서) | |
| 순서·반복 | 2회, r1 = v0.6 → PolyServe, r2 반대. 4 run, 약 5.5시간(설정·엔진 기동 포함) | |
| 드라이버·체인 | `k8s/exp07/run_exp170_hour.sh`(변형 `shift62`), `/home/nxclab/scratch/exp170/run.sh`, 로그 `/home/nxclab/tools/exp170_chain.log` | |
| 원상복구 | 체인이 시작 전에 LWS·게이트웨이 deploy·`llumnix-model` ConfigMap을 JSON으로 저장하고(`/home/nxclab/scratch/exp170/pre/`) 끝에 그대로 다시 적용한다 → 70B 설정을 바이트 단위로 되돌린다. 확인: `switch_model.py --verify llama31-70b-b200-tp2`, `set_api_server_count.py --show`, 게이트웨이 `-v 3` | |

## 판정 (실행 전에 정함)

1. **같은 세션 v0.6 대 PolyServe**: 아홉 지표(모든 도착, admitted, 거절률, 클래스별 admitted·모든 도착)와 token goodput, 지표마다. 문턱 모든 도착·거절률·클래스별 모든 도착 4점, admitted·클래스별 admitted 1점, goodput 5 %, 또는 두 arm 각각의 반복 범위.
2. **유효성**: 모든 run이 엔진 8대로 돌았는가(러너 로그의 cold restart 뒤 8 포트 응답), 클라이언트가 trace를 다 냈는가(분마다 요청 수 ÷ trace, 거절이 아닌 오류 — vLLM\*에서 연결 고갈이 59 % 있었던 함대), 게이트웨이가 `-v 0`인가(run마다 체인이 확인).
3. 같이 적는 것: 엔진 운전점(엔진별 토큰 간격·동시 처리·KV, `engine_work.py`), preemption, PolyServe 거절 이유, 클래스별 TTFT·TBT(`fig_breakdown.py`).
4. **예상**: 70B 섞인 trace에서 v0.6은 PolyServe보다 모든 도착 +27·goodput +36 %였다(EXP-161 대 EXP-168, 세션 다름). 8B 표준 budget에서는 엔진 step이 짧아(chat 50 ms budget 안에 batch가 훨씬 크다)
   PolyServe의 메모리 판정(논리 토큰)이 덜 묶일 수 있어 차이가 줄어들 수 있다. EXP-125의 역전(v0.4)이 v0.6에서도 나는지가 이 실험의 질문이다.

## 결과 (2026-11-21, 4 run 모두 유효)

**유효성**: 네 run 모두 엔진 8대가 run 내내 지표를 보고했다(엔진별 표본 3,307~3,707 / 3,747~3,767, 1 Hz scrape가 부하 중에 빠진 것). 클라이언트는 615,228건을 전부 냈고(분별 요청/trace 0.913~1.100), 거절이 아닌 오류는 run마다 4~6건(400 Bad Request)이다. 게이트웨이는 run마다 `-v 0`. 체인 끝에 LWS·게이트웨이·`llumnix-model`을 저장본으로 되돌렸다 — 저장본과 spec이 같다(replicas만 0), 게이트웨이 `-v 3`·CPU 8·`GOMAXPROCS` 16, ConfigMap `MODEL_ID` 70B, `bin/` 스케줄러 `3251836a`·게이트웨이 `590242b0`. 되돌림 직후의 `switch_model.py --verify`가 "MISMATCH(엔진이 8B를 서비스)"를 냈는데, 이것은 spec이 아니라 **아직 내려가지 않은 8B 엔진 파드에 물어본 것**이다; 곧바로 `gpu_idle.sh start`가 엔진을 0으로 줄였고 Sionna가 돈다. 다음에 엔진을 올릴 때 70B로 뜨는지 `gpu_idle.sh stop`의 엔진 응답 확인에서 본다.

run: v0.6 r1 `261120_0230_exp170hr1_fsg10dl_shift62`, r2 `261120_0603_exp170hr2_fsg10dl_shift62`; PolyServe r1 `261120_0341_exp170hr1_polyservept75_shift62`, r2 `261120_0452_exp170hr2_polyservept75_shift62`. 채점 `analysis_scripts/exp163/observe.py`(ladder95, `FS_SWE_TBT_MS=75 FS_SWE_TTFT_S=7`), 출력 `/home/nxclab/scratch/exp170/obs_<run>.txt`.

### 지표마다 판정 (같은 세션, 각 2회)

| 지표 | v0.6 (r1 / r2, 평균) | PolyServe (r1 / r2, 평균) | 차이 | 문턱 (max(문턱, 두 반복 범위)) | 판정 |
|---|---|---|---|---|---|
| 모든 도착 | 64.71 / 63.03, **63.87** | 72.81 / 74.15, **73.48** | −9.61 | 4 | **PolyServe가 낫다** |
| admitted | 96.48 / 97.03, 96.76 | 95.91 / 96.44, 96.18 | +0.58 | 1 | 차이 없음 |
| 거절률 | 32.76 / 34.90, 33.83 % | 24.00 / 23.03, 23.52 % | +10.31 | 4 | **v0.6이 더 거절한다(나쁨)** |
| chat 모든 도착 | 65.48 / 62.74, 64.11 | 78.61 / 80.51, 79.56 | −15.45 | 4 | **PolyServe가 낫다** |
| chat admitted | 96.19 / 97.16, 96.68 | 94.53 / 95.25, 94.89 | +1.79 | 1 (범위 0.97) | **v0.6이 낫다** |
| deepresearch 모든 도착 | 71.92 / 73.11, 72.52 | 55.11 / 54.52, 54.82 | +17.70 | 4 | **v0.6이 낫다** |
| deepresearch admitted | 97.76 / 97.80, 97.78 | 99.98 / 99.98, 99.98 | −2.20 | 1 | **PolyServe가 낫다** |
| swe 모든 도착 | 48.99 / 47.91, 48.45 | 72.88 / 74.57, 73.73 | −25.28 | 4 | **PolyServe가 낫다** |
| swe admitted | 95.38 / 94.35, 94.87 | 98.72 / 98.92, 98.82 | −3.95 | 1 (범위 1.03) | **PolyServe가 낫다** |
| token goodput | 69,338 / 68,225, 68,782 tok/s | 72,238 / 72,921, 72,580 | −5.2 % | 5 % (범위 1.6 %) | **PolyServe가 낫다**(문턱을 겨우 넘음) |

**결론**: 표준 budget의 8B 함대에서 v0.6은 PolyServe에 **진다** — 모든 도착 −9.6점, 거절률 +10.3점, chat·swe 모든 도착, swe·deepresearch admitted, goodput이 PolyServe가 낫고, v0.6이 나은 것은 deepresearch 모든 도착(+17.7)과 chat admitted(+1.8)이다. admitted는 차이 없음. EXP-125(v0.4, 옛 채점 창, 1회)의 역전(PolyServe 76.0 대 68.2 offered)이 v0.6에서도 같은 방향으로 난다. 두 표는 채점 규칙이 달라 수치끼리 비교하지 않는다. 5분 창으로 보면 v0.6은 두 반복 모두 거의 모든 창에서 두 PolyServe 반복보다 낮다(EXP-171 근거 분석).

### 엔진 운전점 (`analysis_scripts/capacity/engine_work.py`, 분 5~55, 출력 `/home/nxclab/scratch/exp170/engine_work.txt`)

| run | 생성 tok/s | 그중 마감 지킴 | prompt tok/s | prefix 적중 | 엔진당 running | 엔진당 waiting | KV 평균 / p90 | ITL | preemption(엔진 카운터 합) |
|---|---|---|---|---|---|---|---|---|---|
| v0.6 r1 | 73,116 | 0.948 | 257,211 | 0.343 | 384.7 | 27.2 | 0.814 / 0.999 | 42.0 ms | 53,491 |
| v0.6 r2 | 72,078 | 0.947 | 255,856 | 0.344 | 379.5 | 28.2 | 0.817 / 0.999 | 42.1 ms | 53,724 |
| PolyServe r1 | 75,424 | 0.958 | 276,166 | 0.411 | 268.9 | 1.7 | 0.515 / 0.683 | 28.5 ms | 569 |
| PolyServe r2 | 76,268 | 0.956 | 279,419 | 0.415 | 268.6 | 1.5 | 0.504 / 0.679 | 28.1 ms | 298 |

- **엔진당 동시 처리 요청의 최댓값**(`/home/nxclab/scratch/exp170/kv_windows.txt`): v0.6 1,332(p99 1,184), PolyServe 832(p99 724). `--max-num-seqs 4096`에는 어느 쪽도 닿지 않았다. 한도가 1,024였다면 v0.6의 엔진 여섯 대(r1)·다섯 대(r2)가 그 한도에 걸렸을 것이다. FluidServe는 이 값을 읽지 않는다(KV 토큰과 step 시간으로만 판정).
- PolyServe는 KV 0.5 근처, ITL 28 ms에서 생성 토큰을 v0.6보다 3~6 % 더 내고 prefix 적중도 높다(0.41 대 0.34). **8B에서는 v0.6이 KV를 더 채워도 처리량이 늘지 않는다** — 단, 두 정책의 받은 클래스 구성이 달라 정책 간 비교다(EXP-114 §13.42에서 같은 FluidServe 안에서 KV를 0.986 → 0.876으로 낮추었을 때는 tok/s가 7.3 % 줄었다).

### v0.6의 KV — deepresearch-only의 과다 수용과 같은 경로가 8대 전부에서 난다

5분 창별 8대 KV 평균과 preemption(r1 / r2가 같은 모양): 창 4·5(분 20~30)와 10·11(분 50~60)에서 8대 중 7~8대가 KV ≥ 0.973, 엔진당 대기 41~49건, 창마다 preemption 4,600~7,100회. 그 사이 창에서도 2~5대가 0.973 이상. PolyServe는 어느 창에서도 0.973을 넘은 엔진이 없고(최대 0.76), 마지막 부하 구간에만 preemption 7~362회.

메모리 판정이 왜 통과하는가(`/home/nxclab/scratch/exp170/fs_memtest.py`, 인스턴스·초 표본, 두 반복 같음): 논리 KV가 cap_mem의 0.95배 이상인 표본이 75 / 77 %이고, 그 표본에서 projected/논리 KV 중앙값 0.936·0.937, cap_mem/논리 KV 0.951·0.951, 앞으로 h 동안 끝날 것으로 뺀 KV(outflow)/논리 KV 0.081·0.081 → **메모리 판정이 83.4 / 83.0 % 통과한다.** 지금 이미 cap_mem을 넘어 있어도, h 동안 끝날 요청의 KV 8 %를 미리 빼므로 투영값이 cap_mem 아래로 내려가 새 요청을 받는다. 그 8 %는 그 사이에 route되는 요청이 다시 채운다(EXP-171 근거 분석: 투영이 pool의 11 %만큼 낙관적, 두 반복 같음).

### 거절 이유

- **PolyServe**: 거절 판정이 전부 메모리다 — `refused_total{reason=memory}` own tier 2.99M / 2.88M, promotion 2.12M / 2.11M(판정 횟수, 요청 수 아님), 첫 토큰 0~1. PolyServe는 KV를 논리 토큰 + 기대 출력으로 세므로 물리 KV 0.5~0.7에서 멈춘다.
- **v0.6**(인스턴스 하나에 대한 불가 판정 횟수, r1): chat은 첫 토큰 마감 1.16M, gate 0.72M, 메모리 0.54M, KV pace 0.32M; deepresearch는 메모리 0.66M, 첫 토큰 0.37M, gate 0.37M. r2 같은 순서.
- **EXP-171 비판 분석이 새로 찾은 것**: `instcap_sole_block_total{tier=50}`(모든 판정을 통과한 인스턴스가 있었는데 클래스별 인스턴스 상한이 그것을 후보에서 지워 갈 곳이 없어진 스케줄링 주기) 80,084 / 96,811회 대 chat 거절 126,860 / 140,877건. 가장 나쁜 창에서 chat은 8대 중 1.4~4.3대에 묶이고 deepresearch는 8대. 주기 수라 요청 수로 바꿀 수 없고(대기열에서 다시 시도한 주기 포함), 상한을 끈 run은 이 trace에서 없다.

### 남은 질문

인스턴스 상한을 끄면 chat 손실이 줄어드는가(플래그 하나, `FS_INSTANCE_CAP=false`), 그리고 KV 투영을 정직하게 바꾸면서 잃는 수용을 무엇으로 돌려받는가 — EXP-171 분석과 다음 실험 계획에서 다룬다. 절반 budget(논문 구성) 실험은 아직 하지 않았다(디스크 정리 결정 필요: run당 34~41 GB).
