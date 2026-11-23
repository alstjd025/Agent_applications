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

### run 1 — PolyServe r1, 절반 budget (`261122_1804_exp176hr1_polyservept75_shift62h`, 부하 2026-11-23 11:03~12:12) — 1회차 값, 아직 판정하지 않는다

유효성(규칙 (가)): gateway 대체 배치 0건(counter와 로그 모두), scheduler 파드 `7dc2355c`, gateway 파드 `3590c1af` 재시작 0회, 엔진 8/8,
워크로드 slo 2500/25 · 5000/50 · 3500/38(PolyServe 절반 파일), 기동 인자의 tier 표 `25:403,38:464,50:1094` — 전부 맞다. 채점은 절반 budget 여섯 개(`score.py`).

| 지표 | 값 |
|---|---|
| 채점된 도착 수 | 609,123 (미완료 447) |
| 모든 도착 / admitted / 거절률 | 74.07 / 97.03 / 23.61 % |
| token goodput / 엔진이 만든 토큰 전체 | 69,451 / 71,010 tok/s |
| request goodput (ladder95 절반 budget으로 성공한 요청 ÷ 채점 구간 3,588 s) | 125.82 r/s |
| chat (모든 도착 / admitted / 거절률) | 79.68 / 95.98 / 16.96 |
| deepresearch | 46.12 / 99.92 / 53.73 |
| swe | 91.91 / 99.42 / 7.54 |
| preemption / KV 사용률 평균·p90 | 0 / 0.438 · 0.605 |

**메모 (판정 아님, 1회)**: 규칙 (다)의 두 양이 EXP-126 PolyServe와 크게 다르다 — 거절률 23.61 % 대 36.8~37.2 %, request goodput 125.82 대
92.78 / 92.72 r/s(EXP-126의 FluidServe v0.4가 123.67 / 123.61 r/s였다). 2회차가 같은 값이면 "이번 PolyServe는 논문의 절반 budget PolyServe와 다르게
동작한다"가 되고, 그 원인 후보는 세 가지다: ① EXP-126은 옛 gateway(대체 배치 횟수 모름 — 단, 대체 배치는 거절을 줄이는 쪽으로 작용하므로 이번에 거절률이
더 낮은 것을 설명하지 못한다), ② scheduler 바이너리가 다르다(EXP-126 `903889ac`, 이번 `7dc2355c`; 그 사이 EXP-127이 PolyServe의 메모리 판정을 다뤘다),
③ EXP-165가 찾은 hour 드라이버의 tier 표 결함(2026-11-18 이전의 hour 드라이버는 PolyServe tier 표를 기본 파일에서 만들었다; EXP-126 기록은 기동 전 출력으로
25/38/50을 확인했다고 적었으므로 해당하지 않을 수 있다). 2회차 뒤에 run 디렉토리의 scheduler 기동 인자(`scheduler_deploy_*.yaml`)로 확인한다.

### run 2 — 후보 3 r1, 절반 budget (`261122_1919_exp176hr1_fsxc3_shift62h`, 부하 2026-11-23 12:18~13:28) — 1회차 값, 아직 판정하지 않는다

유효성(규칙 (가)): gateway 대체 배치 0건, scheduler 파드 `5d52124e`, gateway 파드 `3590c1af` 재시작 0회, 엔진 8/8, 워크로드 slo 2500/50 · 5000/100 · 3500/25
(FluidServe의 tier key 파일), 기동 인자 `--fluidserve-class-budgets 25:decode:38,50:decode:25,100:decode:50`·`--fluidserve-class-instance-cap false`·
`--fluidserve-kv-price on`, 설정 스크립트의 `verified: every FluidServe flag matches what was applied` — 전부 맞다.

| 지표 | 값 |
|---|---|
| 채점된 도착 수 | 608,852 (미완료 810) |
| 모든 도착 / admitted / 거절률 | 76.92 / 99.73 / 22.76 % |
| token goodput / 엔진이 만든 토큰 전체 | 76,376 / 76,635 tok/s |
| request goodput | 130.65 r/s |
| chat (모든 도착 / admitted / 거절률) | 80.12 / 99.86 / 19.76 |
| deepresearch | 68.10 / 99.51 / 31.20 |
| swe | 75.43 / 99.31 / 24.01 |
| preemption / KV 사용률 평균·p90 | 7,239 / 0.595 · 0.990 |

### run 3 — v0.6 r1, 절반 budget (`261122_2039_exp176hr1_fsg10dl_shift62h`, 부하 2026-11-23 13:38~14:48) — 1회차 값, 아직 판정하지 않는다

유효성(규칙 (가)): gateway 대체 배치 0건, scheduler 파드 `7dc2355c`, gateway 파드 `3590c1af` 재시작 0회, 엔진 8/8, 워크로드 slo 2500/50 · 5000/100 · 3500/25,
기동 인자 `--fluidserve-class-budgets 25:decode:38,50:decode:25,100:decode:50`·`--fluidserve-class-instance-cap true`·KV 가격 플래그 없음,
설정 스크립트의 `verified: every FluidServe flag matches what was applied` — 전부 맞다.

| 지표 | 값 |
|---|---|
| 채점된 도착 수 | 608,787 (미완료 812) |
| 모든 도착 / admitted / 거절률 | 75.94 / 99.38 / 23.48 % |
| token goodput / 엔진이 만든 토큰 전체 | 75,931 / 76,213 tok/s |
| request goodput | 129.02 r/s |
| chat (모든 도착 / admitted / 거절률) | 79.48 / 99.91 / 20.42 |
| deepresearch | 67.20 / 97.64 / 30.82 |
| swe | 72.58 / 99.22 / 26.80 |
| preemption / KV 사용률 평균·p90 | 10,284 / 0.602 · 0.996 |

### run 4 — PolyServe r2, 절반 budget (`261122_2150_exp176hr2_polyservept75_shift62h`, 부하 2026-11-23 14:49~15:58)

유효성(규칙 (가)): gateway 대체 배치 0건, scheduler 파드 `7dc2355c`, gateway 파드 `3590c1af` 재시작 0회, 엔진 8/8, 워크로드 slo 2500/25 · 5000/50 · 3500/38,
기동 인자의 tier 표 `25:403,38:464,50:1094` — 전부 맞다.

| 지표 | r2 | r1 / r2 (평균, 범위) |
|---|---|---|
| 채점된 도착 수 | 608,959 (미완료 468) | |
| 모든 도착 | 75.70 | 74.07 / 75.70 (74.89, 1.63) |
| admitted | 96.65 | 97.03 / 96.65 (96.84, 0.38) |
| 거절률 | 21.62 % | 23.61 / 21.62 (22.62, 1.99) |
| token goodput / 엔진이 만든 토큰 전체 | 70,421 / 72,398 tok/s | goodput 69,451 / 70,421 (69,936) |
| request goodput | 128.65 r/s | 125.82 / 128.65 (127.24) |
| chat (모든 도착 / admitted / 거절률) | 83.00 / 95.53 / 13.09 | 모든 도착 79.68 / 83.00 (범위 3.32) |
| deepresearch | 45.66 / 99.92 / 54.18 | 모든 도착 46.12 / 45.66 |
| swe | 88.56 / 99.39 / 10.89 | 모든 도착 91.91 / 88.56 (범위 3.35) |
| preemption / KV 사용률 평균·p90 | 0 / 0.443 · 0.604 | |

#### 규칙 (다) 판정 — 이번 PolyServe(절반 budget, 2회) 대 EXP-126 PolyServe(2회)

- 거절률: 22.62 % 대 36.8~37.2 %(평균 약 37.0), 차이 **−14.4점** > 문턱 4.6 → **다르다**.
- request goodput: 127.24 대 92.78 / 92.72 r/s, **+37 %** > 5 % → **다르다**.

실행 전 규칙대로 이것을 정책의 차이로 읽지 않고 "측정 경로가 같은지"를 확인했다. 원인 후보 셋:

1. **EXP-165의 tier 표 결함 — EXP-126의 PolyServe arm에 있었다(확인).** EXP-126 체인 로그(`/home/nxclab/tools/exp126.log` 113~120행, PolyServe
   arm으로 전환한 직후)에 scheduler가 받은 tier 표가 `--polyserve-tier-decode-tokens=25:464,50:403,100:1094`로 찍혀 있다. 이 표는 기본 파일
   (`mix_short_m1_balanced.json`, swe 25 / chat 50 / deepresearch 100)에서 만든 것이고, 그 run의 요청은 절반 파일의 tier(chat 25 / deepresearch 50 /
   swe 38)를 달고 있었다. EXP-126 체인 스크립트(`/home/nxclab/tools/exp126_halfslo_hour.sh` 136행)는 `PS_MIX_CONFIG`를 사전 점검용 python 안에서만
   설정했고, 드라이버의 `set_arm`은 그 변수를 지웠다(EXP-165가 찾은 결함). EXP-126 기록 §3의 "tier 경계 25/38/50을 사전 출력으로 확인했다"는 그
   사전 점검의 출력이었고 scheduler가 실제로 받은 값이 아니었다.
   PolyServe 코드에서 이 표는 tier의 출력 길이를 찾는 데만 쓰인다(`polyserve.go`의 KV 투영 `decodeTokens.forTier`, `polyserve_repartition.go`의
   tier별 수요 계산). tier 자체는 요청의 TPOT budget이 정한다. 그래서 EXP-126의 PolyServe는 chat 요청을 swe의 길이 464로, deepresearch 요청을 chat의
   길이 403(실제 클래스 평균 1,094)으로, swe 요청을 표에 없는 tier로 보고 기본값으로 계산했다. 즉 KV 투영과 tier별 서버 배분이 클래스 길이를 잘못
   쓴 상태로 돌았다. 이것이 거절률 차이 −14.4점을 얼마나 설명하는지는 이 자료로는 정하지 못한다(EXP-126의 run 디렉토리가 이 기계에 없다).
2. **바이너리 차이(`903889ac` 대 `7dc2355c`)**: EXP-126 이후 PolyServe 파일(`polyserve*.go`)의 커밋은 `2e752f2`(2026-09-23, 메모리 거절의 두 비교값을
   지표로 내보내는 계측) 하나뿐이다. PolyServe의 결정 코드는 같다고 본다(공유 코드의 변경은 확인하지 않았다).
3. **옛 gateway**: 대체 배치는 admission 판정 없이 받아들이는 쪽이라 거절률을 낮춘다. EXP-126의 거절률이 더 높으므로 이것으로는 설명되지 않는다.

**결론**: 제출 논문의 8B 패널(그림 11·12 (c))의 PolyServe 값은 tier 표 결함이 있는 상태로 잰 값이다. 같은 절반 budget에서 결함 없이 잰 PolyServe는
request goodput 127.24 r/s(2회)로, 논문의 FluidServe v0.4 값(123.67 / 123.61 r/s)과 비슷하다. 논문의 "8B에서 PolyServe 대비 1.33배"는 이 결함 위의
값이다. 같은 결함이 70B 한 시간 trace(EXP-109, 그림 11·12 (a))의 PolyServe에도 있었는지는 EXP-165 기록과 대조해야 한다(EXP-165는 "run_exp109_hour.sh
이후 모든 hour 드라이버"라고 적었다; 70B는 tier 키가 표준 budget과 같은 25/50/100이라 swe만 75 대 25로 어긋났다).

### run 5 — 후보 3 r2, 절반 budget (`261122_2305_exp176hr2_fsxc3_shift62h`, 부하 2026-11-23 16:04~17:14)

유효성(규칙 (가)): gateway 대체 배치 0건, scheduler 파드 `5d52124e`, gateway 파드 `3590c1af` 재시작 0회, 엔진 8/8, 워크로드 slo 2500/50 · 5000/100 · 3500/25,
기동 인자 `--fluidserve-class-budgets 25:decode:38,50:decode:25,100:decode:50` — 전부 맞다.

| 지표 | r2 |
|---|---|
| 채점된 도착 수 | 608,969 (미완료 854) |
| 모든 도착 / admitted / 거절률 | 74.08 / 99.55 / 25.48 % |
| token goodput / 엔진이 만든 토큰 전체 | 74,961 / 75,305 tok/s |
| request goodput | 125.84 r/s |
| chat (모든 도착 / admitted / 거절률) | 74.74 / 99.72 / 25.04 |
| deepresearch | 71.06 / 99.36 / 28.09 |
| swe | 75.80 / 98.99 / 23.37 |
| preemption / KV 사용률 평균·p90 | 9,440 / 0.623 · 0.995 |

#### 규칙 (나) — 후보 3 − PolyServe, 절반 budget (각 2회, 같은 chain; 문턱 = max(4점 / admitted 1점 / 5 %, 두 구성의 반복 범위))

| 지표 | 후보 3 (r1 / r2) | PolyServe (r1 / r2) | 차이 | 문턱 | 판정 |
|---|---|---|---|---|---|
| 모든 도착 | 75.50 (76.92 / 74.08) | 74.88 (74.07 / 75.70) | +0.62 | 4.00 | 차이 없음 |
| admitted | 99.64 (99.73 / 99.55) | 96.84 (97.03 / 96.65) | **+2.79** | 1.00 | 다르다 (후보 3이 높다) |
| 거절률 | 24.12 (22.76 / 25.48) | 22.62 (23.61 / 21.62) | +1.50 | 4.00 | 차이 없음 |
| token goodput | 75,668 | 69,936 | **+8.2 %** | 5 % | 다르다 (후보 3이 높다) |
| 엔진이 만든 토큰 전체 | 75,970 | 71,704 | **+5.95 %** | 5 % | 다르다 (후보 3이 높다) |
| request goodput | 128.24 | 127.23 r/s | +0.8 % | 5 % | 차이 없음 |
| chat 모든 도착 | 77.43 (80.12 / 74.74) | 81.34 (79.68 / 83.00) | −3.91 | 5.38 | 차이 없음 |
| chat admitted | 99.79 | 95.75 | **+4.04** | 1.00 | 다르다 (후보 3이 높다) |
| chat 거절률 | 22.40 | 15.02 | **+7.38** | 5.28 | 다르다 (후보 3이 더 거절) |
| deepresearch 모든 도착 | 69.58 | 45.89 | **+23.69** | 4.00 | 다르다 (후보 3이 높다) |
| deepresearch admitted | 99.44 | 99.92 | −0.48 | 1.00 | 차이 없음 |
| deepresearch 거절률 | 29.65 | 53.95 | **−24.30** | 4.00 | 다르다 (후보 3이 덜 거절) |
| swe 모든 도착 | 75.62 | 90.24 | **−14.62** | 4.00 | 다르다 (PolyServe가 높다) |
| swe admitted | 99.15 | 99.40 | −0.25 | 1.00 | 차이 없음 |
| swe 거절률 | 23.69 | 9.21 | **+14.48** | 4.00 | 다르다 (후보 3이 더 거절) |
| preemption | 7,239 / 9,440 | 0 / 0 | | | 문턱 없음, 방향만 |

**읽기**: 절반 budget에서 후보 3과 PolyServe는 모든 도착·거절률·request goodput이 같다. 후보 3은 받은 요청의 마감을 더 지키고(admitted +2.79, chat admitted
+4.04), 엔진이 만든 토큰과 마감을 지킨 토큰이 더 많다(+6 %, +8 %). 클래스 구성이 정반대다: 후보 3은 deepresearch를 많이 받고 swe·chat을 더 거절하며,
PolyServe는 deepresearch의 54 %를 거절하고 swe를 91 %까지 처리한다. **예측 1(후보 3이 PolyServe보다 모든 도착이 4점 넘게 높다)은 후보 3에 대해 반증됐다.**
v0.6 쪽은 run 6 뒤에 판정한다.

### run 6 — v0.6 r2, 절반 budget (`261123_0025_exp176hr2_fsg10dl_shift62h`, 부하 2026-11-23 17:24~18:34)

유효성(규칙 (가)): gateway 대체 배치 0건, scheduler 파드 `7dc2355c`, gateway 파드 `3590c1af` 재시작 0회, 엔진 8/8, 워크로드 slo 2500/50 · 5000/100 · 3500/25,
기동 인자 `--fluidserve-class-budgets 25:decode:38,50:decode:25,100:decode:50` — 전부 맞다.

| 지표 | r2 |
|---|---|
| 채점된 도착 수 | 608,833 (미완료 901) |
| 모든 도착 / admitted / 거절률 | 74.95 / 99.19 / 24.33 % |
| token goodput / 엔진이 만든 토큰 전체 | 75,297 / 75,666 tok/s |
| request goodput | 127.33 r/s |
| chat (모든 도착 / admitted / 거절률) | 77.72 / 99.82 / 22.12 |
| deepresearch | 67.56 / 97.03 / 29.97 |
| swe | 73.22 / 99.22 / 26.15 |
| preemption / KV 사용률 평균·p90 | 12,213 / 0.622 · 0.997 |

#### 규칙 (나) — v0.6 − PolyServe, 절반 budget (각 2회, 같은 chain)

| 지표 | v0.6 (r1 / r2) | PolyServe | 차이 | 문턱 | 판정 |
|---|---|---|---|---|---|
| 모든 도착 | 75.45 (75.94 / 74.95) | 74.88 | +0.56 | 4.00 | 차이 없음 |
| admitted | 99.29 (99.38 / 99.19) | 96.84 | **+2.44** | 1.00 | 다르다 (v0.6이 높다) |
| 거절률 | 23.90 | 22.62 | +1.29 | 4.00 | 차이 없음 |
| token goodput | 75,614 | 69,936 | **+8.1 %** | 5 % | 다르다 (v0.6이 높다) |
| 엔진이 만든 토큰 전체 | 75,940 | 71,704 | **+5.9 %** | 5 % | 다르다 (v0.6이 높다) |
| request goodput | 128.17 | 127.23 r/s | +0.7 % | 5 % | 차이 없음 |
| chat 모든 도착 / admitted / 거절률 | 78.60 / 99.86 / 21.27 | 81.34 / 95.75 / 15.02 | −2.74 / **+4.11** / **+6.25** | 4 / 1 / 4 | admitted·거절률만 다르다 |
| deepresearch | 67.38 / 97.34 / 30.40 | 45.89 / 99.92 / 53.95 | **+21.49** / **−2.58** / **−23.56** | 4 / 1 / 4 | 셋 다 다르다 |
| swe | 72.90 / 99.22 / 26.47 | 90.24 / 99.40 / 9.21 | **−17.33** / −0.19 / **+17.26** | 4 / 1 / 4 | 모든 도착·거절률이 다르다 |

#### 규칙 (나) — 후보 3 − v0.6, 절반 budget (각 2회, 같은 chain)

| 지표 | 후보 3 | v0.6 | 차이 | 문턱 | 판정 |
|---|---|---|---|---|---|
| 모든 도착 | 75.50 | 75.45 | +0.06 | 4.00 | 차이 없음 |
| admitted | 99.64 | 99.29 | +0.35 | 1.00 | 차이 없음 |
| 거절률 | 24.12 | 23.90 | +0.22 | 4.00 | 차이 없음 |
| token goodput / 엔진 출력 / request goodput | 75,668 / 75,970 / 128.24 | 75,614 / 75,940 / 128.17 | +0.07 % / +0.04 % / +0.05 % | 5 % | 차이 없음 |
| chat 모든 도착 / admitted / 거절률 | 77.43 / 99.79 / 22.40 | 78.60 / 99.86 / 21.27 | −1.18 / −0.07 / +1.13 | 5.38 / 1 / 5.28 | 차이 없음 |
| deepresearch | 69.58 / 99.44 / 29.65 | 67.38 / 97.34 / 30.40 | +2.20 / **+2.10** / −0.75 | 4 / 1 / 4 | admitted만 다르다 (후보 3이 높다) |
| swe | 75.62 / 99.15 / 23.69 | 72.90 / 99.22 / 26.47 | +2.71 / −0.07 / −2.78 | 4 / 1 / 4 | 차이 없음 |
| preemption | 7,239 / 9,440 (평균 8,340) | 10,284 / 12,213 (평균 11,249) | −2,909 (−26 %) | 없음 | 두 반복 모두 후보 3이 적다 — 방향만 적는다 |

#### 예측 판정 (절반 budget)

- **예측 1(v0.6과 후보 3이 PolyServe보다 모든 도착이 4점 넘게 높다) — 반증.** 차이는 +0.56, +0.62. 두 FluidServe 구성이 PolyServe보다 나은 지표는 admitted
  (+2.4~2.8), token goodput(+8 %), 엔진 출력(+6 %), chat admitted(+4)이고, 마감을 지킨 요청 수는 같다. 클래스 구성은 정반대다(FluidServe는 deepresearch를
  +21~24점 더 처리하고 swe를 −15~17점 덜 처리한다).
- **예측 2(후보 3과 v0.6은 어느 지표도 다르지 않다) — 한 지표에서 반증.** deepresearch admitted만 후보 3이 2.10점 높다(v0.6 97.34, 후보 3 99.44). 우려했던
  방향(후보 3의 admitted가 낮다)은 나타나지 않았다. preemption은 후보 3이 26 % 적다(문턱 없음).

#### 절반 budget 종합 (여섯 run, 2026-11-23)

1. 결함 없이 잰 절반 budget에서 FluidServe(v0.6, 후보 3)는 PolyServe와 **끝까지 처리한 요청 수가 같다**(모든 도착 75.5 대 74.9, request goodput 128 대 127 r/s).
   논문 그림 12 (c)의 1.33배는 PolyServe의 tier 표 결함(run 4 절) 위의 값이었다.
2. FluidServe가 나은 것은 **받은 요청의 마감**(admitted 99.3~99.6 대 96.8, 특히 chat 99.8 대 95.8)과 **처리량**(마감을 지킨 토큰 +8 %)이다. PolyServe는
   chat을 덜 거절하지만 받은 chat의 4 %가 마감을 놓친다.
3. 클래스 선택이 다르다: PolyServe는 deepresearch의 54 %를 거절하고 swe를 90 % 처리하며, FluidServe는 세 클래스의 거절률이 21~30 %로 고르다.
4. 절반 budget에서는 후보 3과 v0.6이 사실상 같다(deepresearch admitted 하나, preemption 방향). 표준 budget 8B(EXP-174)에서 후보 3이 v0.6보다 모든 도착 +20.4점
   높았던 차이는 KV가 먼저 한계인 조건에서만 나타난다.

### run 7 — PolyServe r1, 더 빡빡한 budget (`261123_0136_exp176hr1_polyservept75_shift62t`, 부하 2026-11-23 18:45~19:55) — 1회차 값, 아직 판정하지 않는다

유효성(규칙 (가)): gateway 대체 배치 counter **1건**(받아들인 요청의 0.1 % 기준 약 420건보다 훨씬 작다 → 유효), scheduler 파드 `7dc2355c`, gateway 파드 `3590c1af`
재시작 0회, 엔진 8/8, 워크로드 slo 2500/15 · 5000/35 · 3500/25, 기동 인자의 tier 표 `15:403,25:464,35:1094` — 맞다. 채점은 더 빡빡한 budget 여섯 개
(`FS_CHAT_TBT_MS=15 FS_DR_TBT_MS=35 FS_SWE_TBT_MS=25`, 첫 토큰 2.5 / 5 / 3.5 s).

| 지표 | 값 |
|---|---|
| 채점된 도착 수 | 609,434 (미완료 362) |
| 모든 도착 / admitted / 거절률 | 64.35 / 94.05 / 31.54 % |
| token goodput / 엔진이 만든 토큰 전체 | 62,608 / 65,412 tok/s |
| request goodput | 109.29 r/s |
| chat (모든 도착 / admitted / 거절률) | 67.37 / 91.91 / 26.69 |
| deepresearch | 48.44 / 99.52 / 51.22 |
| swe | 75.39 / 98.62 / 23.54 |
| preemption / KV 사용률 평균·p90 | 0 / 0.278 · 0.588 |

**체인의 기록 파일 이름 충돌(2026-11-23 20:00 발견)**: 체인이 run마다 쓰는 gateway 로그·지표 사본(`/home/nxclab/scratch/exp176/gwlog_<rep>_<arm>.log.gz`,
`gwmetrics_<rep>_<arm>.txt`)의 이름에 budget 변형이 없어서, run 7이 run 1(PolyServe r1, 절반 budget)의 두 파일을 덮어썼다. **잃은 것은 run 1의 gateway
로그 원문 하나**다. run 1의 대체 배치 counter는 체인 로그(0건, 로그 줄 0개)와 run 디렉토리의 `server_metrics/gateway.jsonl`(마지막 값 0)에 남아 있어
유효성 판정은 바뀌지 않는다. 남은 절반 budget 파일 다섯 쌍은 덮어쓰이기 전에 `_shift62h`를 붙인 사본으로 복사했고, run 7의 파일도 `_shift62t` 사본을
만들었다. 체인 스크립트는 실행 중이라 고치지 않았다(남은 run 8~12는 각각 절반 budget의 원래 이름 파일을 덮어쓰지만 그 내용은 사본에 있다).

### run 8 — 후보 3 r1, 더 빡빡한 budget (`261123_0301_exp176hr1_fsxc3_shift62t`, 부하 2026-11-23 20:01~21:10) — 1회차 값, 아직 판정하지 않는다

유효성(규칙 (가)): gateway 대체 배치 0건, scheduler 파드 `5d52124e`, gateway 파드 `3590c1af` 재시작 0회, 엔진 8/8, 워크로드 slo 2500/50 · 5000/100 · 3500/25,
기동 인자 `--fluidserve-class-budgets 25:decode:25,50:decode:15,100:decode:35` — 맞다. gateway 사본은 `_shift62t` 이름으로도 남겼다.

| 지표 | 값 |
|---|---|
| 채점된 도착 수 | 609,051 (미완료 648) |
| 모든 도착 / admitted / 거절률 | 74.17 / 99.95 / 25.72 % |
| token goodput / 엔진이 만든 토큰 전체 | 74,232 / 74,293 tok/s |
| request goodput | 125.94 r/s |
| chat (모든 도착 / admitted / 거절률) | 75.33 / 99.98 / 24.66 |
| deepresearch | 68.79 / 99.86 / 30.79 |
| swe | 77.19 / 99.93 / 22.74 |
| preemption / KV 사용률 평균·p90 | 1,315 / 0.443 · 0.954 |

판정 조건 하나만으로 후보가 불가가 된 평가: 토큰 간격(gate) chat 1,398,864 · deepresearch 312,783 · swe 94,027, 처리 중 요청 보호 70,930, 첫 토큰 마감
50,652, 메모리 3,644, 가격 거의 0. 이 budget에서는 KV가 거의 한계가 되지 않고 토큰 간격이 거의 모든 거절을 정한다.
