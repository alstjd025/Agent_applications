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
