# EXP-176 — 8B 함대에서 논문의 절반 budget과 그보다 더 빡빡한 budget으로 후보 3·v0.6·PolyServe를 잰다 (2026-11-23)

**사용자 지시(2026-11-23)**: "그러면 후보3으로 8B에서 SLO를 논문 측정으로도 해보자", 그리고 구성(PolyServe·후보 3·v0.6 각 2회)에 동의한 뒤
"절반 budget이랑, budget을 더 줄여서 chat는 2.5s, 토큰당 15ms, deep research는 5s, 토큰당 35ms, swe budget은 3.5s 토큰당 25ms으로도 해보자."

**물음**: 표준 budget(chat 5 s / 50 ms, deepresearch 10 s / 100 ms, swe 7 s / 75 ms)의 8B 함대에서는 v0.6이 PolyServe에 지고(모든 도착
63.53 대 77.95, EXP-174), 후보 3은 PolyServe를 앞서지만 admitted가 낮다(83.94 / 95.90 대 77.95 / 99.13). 그 함대에서는 엔진의 토큰 간격이
budget에 닿기 전에 KV pool이 먼저 차기 때문이다(EXP-121의 기록: 8B 엔진에는 표준 per-token budget이 느슨해서 KV가 먼저 한계가 된다).
제출 논문은 8B 함대에서 모든 budget을 절반으로 두었고 거기서 FluidServe v0.4가 PolyServe를 크게 앞섰다(EXP-126). 이 실험은
① 논문과 같은 절반 budget에서, ② per-token budget을 그보다 더 줄인 조건에서, **후보 3과 v0.6이 PolyServe보다 앞서는지, 후보 3과 v0.6이
서로 다른지**를 새 gateway(scheduler 연결 재사용)로 잰다.

## 1. 두 budget 조건

| 클래스 | 표준 (EXP-170~174) | **절반 (변형 `shift62h`, 논문 8B = EXP-126)** | **더 빡빡함 (변형 `shift62t`)** |
|---|---|---|---|
| chat | 5 s / 50 ms | 2.5 s / 25 ms | 2.5 s / 15 ms |
| deepresearch | 10 s / 100 ms | 5 s / 50 ms | 5 s / 35 ms |
| swe | 7 s / 75 ms | 3.5 s / 38 ms | 3.5 s / 25 ms |

첫 토큰 budget은 두 조건이 같고 per-token budget만 다르다. swe의 절반 값이 37.5가 아니라 38 ms인 것은 scheduler의
`--fluidserve-class-budgets`가 정수만 받기 때문이다(EXP-126과 같은 값).

엔진이 낼 수 있는 토큰 간격의 범위(8B 프로파일 `llama31-8b-b200-tp1`의 decode step 모델): decode만 하는 step은 4.29 ms + 요청당
0.031 ms + KV 토큰당 6.8×10⁻⁶ ms이므로 15 ms는 KV가 적을 때 요청 약 300개, KV 100만 토큰이면 요청 약 100개까지 들어가는 시간이다.
prefill chunk를 실은 step은 1,024 토큰 15.9 ms, 4,096 토큰 56.9 ms, 8,192 토큰 117.3 ms이므로, 15 ms 조건에서는 prefill이 섞인 step
하나가 그 자체로 chat의 토큰당 budget을 넘는다. ladder95는 i번째 토큰의 마감을 "첫 토큰 budget + i × 토큰당 budget"으로 두므로
그 초과분은 첫 토큰 budget에서 남은 시간(budget 여유)으로 흡수되어야 한다.

## 2. 구성

| 항목 | 값 |
|---|---|
| 함대 | Llama-3.1-8B × 8, TP=1, `--api-server-count 4`, `--max-num-seqs 4096`, gateway `-v 0`·CPU 24 (EXP-170~174와 같음) |
| trace | `dyn60_shift_m2Am1B_b1045_x620.csv`(70B 한 시간 trace를 6.20배로 늘린 것, 도착 615,228건), 부하 생성 36 프로세스 |
| gateway | `gateway-connreuse2` `3590c1af`(scheduler 연결 재사용, EXP-174) — 모든 run 공통 |
| scheduler 바이너리 | PolyServe·v0.6 `scheduler-gateutil` `7dc2355c`, 후보 3 `scheduler-exp172-c3` `5d52124e` (EXP-174와 같은 파일) |
| arm | **PolyServe** `polyservept75`, **v0.6** `fsg10dl`, **후보 3** `fsxc3`(v0.6 + `FS_INSTANCE_CAP=false` + `FS_KV_PRICE=on`, arm 파일 `/home/nxclab/scratch/exp172/arms/fsxc3.env`) |
| FluidServe 계열 워크로드 | 두 조건 모두 `mix_dyn60_shift_m2Am1B_b1045_x620_ftc2500d5000s3500.json`(EXP-123에서 만든 파일, md5 `f514823e…`, tbt_ms는 class를 부르는 tier key 50/100/25) |
| FluidServe per-token budget | 절반 `--fluidserve-class-budgets 25:decode:38,50:decode:25,100:decode:50`, 더 빡빡함 `25:decode:25,50:decode:15,100:decode:35` (`FS_*_TBT_MS`를 arm 설정 뒤에 export한다 — v0.6 계열 arm이 `FS_SWE_TBT_MS=75`를 export하므로 순서가 바뀌면 swe가 75 ms로 들어간다) |
| PolyServe 워크로드 | 절반 `mix_dyn60_shift_m2Am1B_b1045_x620_c25d50s38ftc2500d5000s3500fair.json`(EXP-126의 파일, md5 `5dd51c01…`), 더 빡빡함 **새 파일** `mix_dyn60_shift_m2Am1B_b1045_x620_c15d35s25ftc2500d5000s3500fair.json`(md5 `11a9fda4…`, 절반 파일에서 tbt_ms 세 값과 `_comment`만 바꿈) |
| PolyServe tier 표 | 워크로드 파일에서 만든다: 절반 `25:403,38:464,50:1094`, 더 빡빡함 `15:403,25:464,35:1094`(표준은 `50:403,75:464,100:1094`). tier 경계가 budget을 따라 움직이므로 PolyServe는 채점뿐 아니라 class 파티션 자체가 표준 조건과 다르다 |
| 채점 | ladder95, 그 조건의 budget 여섯 개를 환경변수로 준다: `FS_CHAT_TTFT_S=2.5 FS_DR_TTFT_S=5 FS_SWE_TTFT_S=3.5` + 절반 `FS_CHAT_TBT_MS=25 FS_DR_TBT_MS=50 FS_SWE_TBT_MS=38` / 더 빡빡함 `15 / 35 / 25`. **대시보드는 표준 budget으로 채점하므로 대시보드 점수를 쓰지 않는다.** 채점기 `/home/nxclab/scratch/exp176/score.py`(대시보드의 채점 함수를 그 환경으로 부른다) |
| 드라이버 | `k8s/exp07/run_exp176_hour.sh` — `run_exp172_hour.sh`의 사본. 차이: 변형 `shift62h`·`shift62t` 추가(trace는 `shift62`와 같음, 워크로드·budget은 위 표), `FS_CHAT_TBT_MS`·`FS_DR_TBT_MS`를 arm 사이 초기화 목록에 추가, 이름(`exp176`) |
| 체인 | `/home/nxclab/scratch/exp176/run.sh`(EXP-174 체인의 사본 + run마다 scheduler 기동 인자의 budget 문자열과 PolyServe tier 표를 기록), 로그 `/home/nxclab/tools/exp176_chain.log`, session prefix `exp176h` |

**순서 (각 약 71분, 전체 약 15시간)** — 절반 budget 여섯 run을 먼저, 그 다음 더 빡빡한 budget 여섯 run. 각 조건 안에서는 반복이 바깥 루프다:
PolyServe r1 → 후보 3 r1 → v0.6 r1 → PolyServe r2 → 후보 3 r2 → v0.6 r2. 비교는 같은 budget 조건 안에서만 하므로 조건을 묶어 돌린다.

v0.6은 절반·더 빡빡한 budget에서 한 번도 잰 적이 없으므로(논문 8B는 v0.4) 같은 세션 비교를 위한 재측정이 아니라 새 측정이다.

## 3. 판정 규칙 (실행 전에 쓴다)

**(가) run 유효성** — 다음 중 하나라도 어긋나면 그 run은 무효이고 다시 잰다.
- gateway의 대체 배치 counter(`gateway_scheduling_fallback_total`, scheduler 호출이 실패해 admission 판정 없이 round-robin으로 보낸 요청 수)가
  받아들인 요청의 0.1 % 이하. 2,000건을 넘으면 체인을 멈춘다(EXP-174와 같은 규칙).
- scheduler·gateway 파드 안 `md5sum /proc/1/exe`가 큐에 적은 값. scheduler 재시작 0회, 엔진 8대가 끝까지 지표를 보고.
- run_config의 워크로드 slo가 위 표의 값(FluidServe 계열 2500/50·5000/100·3500/25, PolyServe 절반 2500/25·5000/50·3500/38, 더 빡빡함 2500/15·5000/35·3500/25).
- FluidServe 계열: scheduler 기동 줄(`verified:`)과 기동 인자의 `--fluidserve-class-budgets`가 그 조건의 문자열. PolyServe: 기동 인자의 tier 표가 위 표의 값.

**(나) 같은 budget 조건 안의 비교 세 개** — 각 조건(절반, 더 빡빡함)마다 따로:
후보 3 − PolyServe, 후보 3 − v0.6, v0.6 − PolyServe. 모든 지표를 따로 판정한다(모든 도착, admitted, 거절률, token goodput, 엔진이 만든 토큰 전체,
클래스별 모든 도착 / admitted / 거절률). 문턱은 같은 chain의 값: 모든 도착·거절률·클래스별 모든 도착·클래스별 거절률 4점, admitted·클래스별 admitted 1점,
goodput·엔진 출력 5 %; 두 구성 각각의 반복 범위가 더 크면 그 범위. preemption 횟수는 같은 chain 안에서만 적고, 미리 정한 문턱이 없으므로 방향만 적는다.

**(다) 논문 값과의 대조 (절반 budget만, 참고)** — 이번 PolyServe(절반) − EXP-126 PolyServe. 비교할 수 있는 양은 둘이다.
① 거절률(채점 규칙과 무관): EXP-126 36.8~37.2 %. ② request goodput(ladder95 절반 budget으로 성공한 요청 수 ÷ 채점 구간 초): EXP-126 92.78 / 92.72 r/s
(논문 그림 12의 값). 문턱은 세션 간 값(거절률 4.6점, request goodput 5 %). EXP-126은 옛 gateway로 잰 것이고 그 run의 대체 배치 횟수는 알 수 없으므로,
차이가 나와도 정책의 차이로 읽지 않고 "측정 경로가 같은지 확인되지 않음"으로 적는다. EXP-126의 달성률 표(FluidServe 73.5, PolyServe 44.6~45.0)는
평균 규칙으로 센 값이므로 대조하지 않는다. FluidServe는 논문이 v0.4이므로 대조하지 않는다.

**(라) 두 budget 조건 사이** — 채점 규칙이 다르므로 값을 한 표에 놓거나 빼지 않는다. 각 조건 안의 순서(어느 구성이 어느 지표에서 앞서는지)만 나란히 적는다.

## 4. 예측 (실행 전)

1. **절반 budget에서 v0.6과 후보 3은 둘 다 PolyServe보다 모든 도착과 admitted가 4점 넘게 높다.** 근거: per-token budget을 절반으로 두면 8B 엔진에서도
   토큰 간격이 KV보다 먼저 한계가 되고(EXP-121), 그 조건에서 FluidServe v0.4가 PolyServe를 모든 도착 약 29점 앞섰다(EXP-126, 평균 규칙). 70B에서
   v0.6이 PolyServe를 앞서는 것도 토큰 간격이 먼저 한계가 되는 함대에서다. PolyServe는 tier 경계가 25 / 38 / 50 ms로 13·12 ms 간격까지 좁아져
   파티션이 더 어렵다(EXP-126 §3). **반증**: 절반 budget에서 v0.6이나 후보 3의 모든 도착 평균이 PolyServe보다 4점 넘게 높지 않으면 이 예측은 틀린 것이다.
2. **절반 budget에서 후보 3과 v0.6은 어느 지표도 다르지 않다.** 근거: 70B(토큰 간격이 먼저 한계인 함대)에서 KV 가격이 유일한 실패 조건이었던 평가는 run당
   511~739회뿐이었고 두 구성은 어느 지표도 다르지 않았다(EXP-175). KV가 덜 차면 KV 가격 λ는 거의 0에 머문다. **반증**: 어느 지표든 문턱을 넘는 차이.
   특히 후보 3의 admitted가 v0.6보다 1점 넘게 낮으면 표준 budget 8B의 약점(KV를 넘겨 preemption이 늘어나는 것)이 절반 budget에서도 남는다는 뜻이다.
3. **더 빡빡한 budget에서는 세 구성 모두 거절률이 절반 budget보다 크게 오르고, v0.6과 후보 3은 여전히 PolyServe보다 모든 도착이 높다.** 근거는 1과 같고,
   PolyServe의 tier 경계는 15 / 25 / 35 ms로 10 ms 간격이 된다. 불확실성: chat 15 ms에서는 prefill이 섞인 step 하나(1,024 토큰 15.9 ms)가 budget을
   넘으므로, FluidServe의 판정이 chat을 거의 다 거절하거나, 받은 chat의 마감 위반이 늘어 admitted가 99 아래로 내려갈 수 있다. **반증**: 더 빡빡한 budget에서
   v0.6이나 후보 3의 모든 도착이 PolyServe보다 4점 넘게 높지 않다.

## 5. 한계 (실행 전에 알고 있는 것)

- 구성마다 2회, trace 하나, 모델 하나. 1~3점 차이는 이 반복 수로 가릴 수 없다.
- 더 빡빡한 budget은 사용자가 정한 값이고 SLO를 정하는 외부 근거(논문 Table 2의 출처)가 없다. 논문에 쓸 때는 budget sensitivity로만 쓸 수 있다.
- 절반 budget의 논문 값(EXP-126)은 옛 gateway, FluidServe v0.4로 잰 것이다.

## 6. 결과

(run이 끝난 뒤에 적는다)
