# EXP-175 — 8B 함대에서 PolyServe를 앞선 후보 3(인스턴스별 KV 가격)이 70B 함대에서도 앞서는가 (2026-11-22)

**사용자 지시(2026-11-22)**: "후보3과 polyserve을 70B에서도 돌려봐", "후보3이 만약 70B에서도 앞설지?"

**물음**: EXP-174(게이트웨이가 스케줄러 연결을 재사용하게 고친 뒤 8B 함대를 다시 잰 실험)에서 후보 3은 8B 함대 한 시간 trace에서 PolyServe보다 모든 도착 +5.99점, 거절률 −9.00점, goodput +10.3 %였다(각 2회, 같은 체인; admitted는 95.90 대 99.13으로 후보 3이 낮다). 후보 3을 **70B 함대(Llama-3.1-70B × 4대, TP=2)의 논문 한 시간 trace**에 그대로 가져가면 ① PolyServe보다 앞서는가, ② 지금 기준선인 FluidServe v0.6보다 어느 지표도 나빠지지 않는가(EXP-172에서 후보 3을 만들 때 실행 전에 적어 둔 반증 조건 R4).

## 1. 후보 3이 무엇인가 (EXP-172에서 2026-11-21에 만든 것)

FluidServe v0.6에서 두 가지를 바꾼 것이다.

1. **클래스별 인스턴스 상한을 끈다**(`FS_INSTANCE_CAP=false`, 플래그 `--fluidserve-class-instance-cap=false`). 이 상한은 클래스마다 수요 추정으로 "그 클래스가 토큰 간격 판정의 기준이 될 수 있는 인스턴스 수"를 제한하는 장치다.
2. **인스턴스마다 KV 가격 λ를 둔다**(`FS_KV_PRICE=on`, 플래그 `--fluidserve-kv-price=on`, 바이너리 `scheduler-exp172-c3` `5d52124e`). 부족도 u = (물리 KV 사용량 + 아직 계산되지 않은 prefill 토큰) ÷ 물리 KV pool. u가 1을 넘으면 vLLM이 preemption을 하거나 prompt를 대기시키는 조건이다. 엔진 상태가 갱신될 때마다 λ ← λ·exp((u−1)·Δt/H)(H = horizon h × 그 인스턴스에서 잰 step 시간; 새 조정 상수 없음), u가 1 아래로 내려가면 λ = 0. 요청의 비용 c = (prefix hit을 뺀 prompt 토큰) × E[L] + E[L²]/2(L = 그 클래스의 출력 길이 분포; KV·토큰·step 단위). v0.6의 판정에 더해 **λ·c ≤ 1**이어야 그 인스턴스로 route한다. 요청 하나의 가치는 클래스와 무관하게 1이다(채점 지표가 요청 수이므로).

이론 근거는 network utility maximization의 dual decomposition(Kelly 1998, Low–Lapsley 1999)이다. 만든 이유: EXP-170에서 v0.6이 8B 함대에서 PolyServe에 졌고(모든 도착 63.87 대 73.48), EXP-171이 원인 후보로 "KV가 부족할 때 어느 클래스에 KV를 줄지 정하는 규칙이 없다"를 꼽았기 때문이다.

**이름 주의**: `project/versions/README.md`의 "v0.6-후보3"(arm `fsg10`, 판정선 1.0)은 다른 것이다. 이 문서의 "후보 3"은 EXP-172의 세 번째 후보(arm `fsxc3`)다.

## 2. 구성

70B 함대, EXP-161·168(논문의 섞인 한 시간 trace에서 v0.6과 PolyServe를 잰 실험)과 같은 trace·워크로드 설정·부하 생성 프로세스 수.

| 항목 | 값 |
|---|---|
| 함대 | Llama-3.1-70B × 4대, TP=2 (`PROFILE_DIR=llama31-70b-b200-tp2`) |
| trace | `dyn60_shift_m2Am1B_b1045.csv`(드라이버 변형 `shift`), 채점 대상 도착 약 98k건, 부하 생성 12 프로세스 |
| 워크로드 설정 | FluidServe 계열 `mix_dyn60_shift_m2Am1B_b1045_t75.json`, PolyServe `mix_dyn60_shift_m2Am1B_b1045_t75fair.json` (EXP-161·168과 같은 파일, 고치지 않음) |
| 게이트웨이 | **`gateway-connreuse2` `3590c1af`**(스케줄러 연결 재사용, EXP-174) — 모든 arm 공통. `-v 3`, CPU 8 등 70B의 게이트웨이 설정은 그대로 |
| 드라이버 | `k8s/exp07/run_exp175_hour.sh` — `run_exp172_hour.sh`의 사본. 동작 차이 하나: 변형 `shift`에서 `fsx*` arm이 v0.6과 같은 `…_t75.json`을 읽는다(원본은 `fsx*`를 8B 변형에서만 정의해 `shift`에서는 fail-closed 검사가 중단시킨다). job·session·로그 이름만 `exp175`로 바꿨다 |
| 체인 | `/home/nxclab/scratch/exp175/run.sh`, 로그 `/home/nxclab/tools/exp175_chain.log`, session prefix `exp175h` |

| 순서 | arm | 무엇인가 | 스케줄러 바이너리 | 반복 |
|---|---|---|---|---|
| 1 | `polyservept75` | PolyServe (EXP-168과 같은 구성) | `scheduler-gateutil` `7dc2355c` | r1 |
| 2 | `fsxc3` | 후보 3 (v0.6 + 상한 끔 + KV 가격) | `scheduler-exp172-c3` `5d52124e` | r1 |
| 3 | `fsxc3` | | | r2 |
| 4 | `polyservept75` | | | r2 |
| 5 | `fsxnocap` | v0.6에서 **상한만 끈 것**(KV 가격 없음) | `scheduler-gateutil` `7dc2355c` | r1 |
| 6 | `fsxnocap` | | | r2 |

- **v0.6은 다시 돌리지 않는다**(2026-11-18 사용자 결정: 같은 세션 비교만을 위한 FluidServe 재측정을 하지 않는다). 기록된 EXP-161의 2회를 세션 간 문턱으로 비교한다. 게이트웨이 바이너리가 EXP-161·168(`05745270`)과 다르다는 점은 1·4번 PolyServe가 EXP-168 PolyServe와 같은지로 확인한다(§3 (나)).
- **5·6번을 넣은 이유**: 후보 3은 v0.6에서 두 가지(상한 끔, KV 가격)를 바꾼 것이라, 후보 3과 v0.6이 다르게 나와도 어느 쪽 때문인지 알 수 없다. v0.6에서 상한만 끈 구성은 70B 한 시간 trace에서 잰 적이 없다(제출 논문 구성으로만 EXP-135·137·138에서 쟀고 그때는 상한을 꺼도 85.6~85.8로 차이가 없었다). 사용자가 요청한 비교(1~4번)가 먼저 끝나도록 뒤에 둔다.

기록된 기준 값(ladder95, 70B 섞인 한 시간 trace, 대시보드 채점):

| run | 모든 도착 | admitted | 거절률 % | goodput tok/s | chat 전체/admitted/거절 | deepresearch | swe |
|---|---|---|---|---|---|---|---|
| v0.6 EXP-161 r1 | 80.78 | 99.92 | 18.94 | 11,767 | 85.64 / 99.99 / 14.30 | 68.52 / 99.85 / 30.71 | 76.63 / 99.61 / 22.96 |
| v0.6 EXP-161 r2 | 80.60 | 99.87 | 19.07 | 11,760 | 85.22 / 99.99 / 14.71 | 69.86 / 99.87 / 29.35 | 75.16 / 99.22 / 24.13 |
| PolyServe EXP-168 r1 | 53.37 | 65.18 | 17.93 | 8,637 | 51.30 / 54.21 / 5.22 | 43.07 / 99.07 / 56.27 | 80.66 / 99.42 / 18.80 |
| PolyServe EXP-168 r2 | 54.04 | 65.98 | 17.85 | 8,669 | 52.41 / 55.30 / 5.04 | 42.85 / 99.27 / 56.53 | 80.65 / 99.29 / 18.67 |

## 3. 판정 규칙 (실행 전에 쓴다)

지표는 전부 따로 판정한다: 전체와 클래스별(chat·deepresearch·swe) 모든 도착, admitted, 거절률, 그리고 goodput. preemption 수와 KV 사용률은 같은 체인 안에서만 비교한다.

**(가) run이 유효한가 — run마다**: `gateway_scheduling_fallback_total`(게이트웨이가 스케줄러 호출 실패 때 admission 판정 없이 round-robin으로 보낸 횟수)이 받아들인 요청의 0.1 % 이하(약 80건). 스케줄러·게이트웨이 파드의 `md5sum /proc/1/exe`가 표의 값. 워크로드 확인 줄이 FluidServe 계열 `7000 25 None dyn60_shift_m2Am1B_b1045.csv 12`, PolyServe `7000 75 None … 12`. PolyServe tier 표가 EXP-168과 같은 `50:428,75:494,100:985`. 엔진 4대.

**(나) 측정 경로가 EXP-161·168과 같은가**: 이 체인의 PolyServe(2회) − EXP-168 PolyServe(2회). 세션 간 문턱 — 모든 도착·거절률·클래스별 모든 도착 max(4.6, 두 쪽 반복 범위), admitted·클래스별 admitted max(1, 범위), goodput max(5 %, 범위). **예측: 어느 지표도 다르지 않다**(옛 게이트웨이에서도 70B의 대체 배치는 0이었다). 다르면 게이트웨이 수정이나 8B→70B 복구 과정이 70B 동작을 바꾼 것이므로, 기록된 v0.6과의 비교((라))는 하지 않고 원인을 먼저 찾는다.

**(다) 후보 3 대 PolyServe**(같은 체인, 각 2회): 평균 차이 > max(문턱, 두 쪽 반복 범위)인 지표만 "다르다". 문턱은 한 시간 trace의 같은 체인 값 — 모든 도착·거절률·클래스별 모든 도착 4점, admitted·클래스별 admitted 1점, goodput 5 %.

**(라) 후보 3 대 v0.6(EXP-161)**: (나)와 같은 세션 간 문턱. **반증 조건 R4(EXP-172에서 적은 것)**: 어느 지표든 후보 3이 v0.6보다 문턱을 넘게 나쁘면, 후보 3은 "8B에서 좋아지고 70B에서 손해가 없는" 구성이 아니다.

**(마) 무엇 때문인가**: `fsxnocap` − v0.6(EXP-161)(세션 간 문턱) = 상한을 끈 효과. 후보 3 − `fsxnocap`(같은 체인 문턱) = KV 가격의 효과.

## 4. 예측 (실행 전)

1. **후보 3은 70B에서도 PolyServe를 앞선다 — 다만 그 차이는 후보 3이 새로 만든 것이 아니라 v0.6이 이미 가진 차이다.** 기록된 v0.6은 PolyServe보다 모든 도착이 27점 높다(80.69 대 53.70). 그 차이는 PolyServe가 chat을 받은 뒤 SLO를 못 지키는 데서 나온다(chat admitted 54.75). 후보 3의 KV 가격은 u > 1일 때만 0이 아닌데, EXP-172의 오프라인 추정으로 70B 섞인 trace에서 가격이 거절하는 도착은 deepresearch의 0.35~0.46 %, chat·swe는 그보다 적다. 그래서 70B에서 후보 3은 거의 "v0.6에서 상한을 끈 것"으로 동작할 것이다. 예측 값: 모든 도착 77~83, admitted 99 이상, 거절률 17~22 %.
2. **후보 3 대 v0.6: 전체 지표는 "차이 없음".** 위험은 상한을 끈 쪽에 있다. 상한이 없으면 chat이 deepresearch·swe가 모인 인스턴스로도 route될 수 있고, 그 인스턴스의 허용 토큰 간격이 chat budget 50 ms로 내려가 deepresearch·swe를 덜 받게 된다. 제출 논문 구성의 한 시간 trace에서는 상한을 꺼도 차이가 없었지만(85.6~85.8), 정적 35 req/s에서는 상한이 있는 쪽이 10~12점 **낮았다**(EXP-153의 제거 장부). 따라서 클래스별 모든 도착은 어느 방향으로든 문턱(4.6점) 가까이 움직일 수 있다고 본다 — deepresearch·swe가 내려가는 방향이면 R4에 걸린다.
3. **preemption**: 70B v0.6은 run당 1,612·1,728회였다(EXP-161; 세션이 달라 직접 비교하지 않는다). 같은 체인 안에서 후보 3이 `fsxnocap`보다 적으면 가격이 70B에서도 동작한 것이고, 같으면 70B에서는 가격이 결정에 거의 개입하지 않은 것이다.
4. **(나)의 대조**: PolyServe는 EXP-168과 모든 지표에서 차이 없음.

## 5. 결과

(run이 끝난 뒤에 적는다)

### run 1 — PolyServe r1 (`261122_0648_exp175hr1_polyservept75_shift`, 부하 2026-11-22 23:53~2026-11-23 00:56) — 1회차 값, 아직 판정하지 않는다

유효성(규칙 (가)): gateway의 대체 배치 counter 0건(gateway 로그의 `cannot assign` 0줄), scheduler 파드 `7dc2355c`, gateway 파드 `3590c1af`(재시작 0),
workload 줄 `7000 75 None dyn60_shift_m2Am1B_b1045.csv 12`, tier table `50:428,75:494,100:985`, 엔진 4대 모두 지표 보고. 유효하다.

| 지표 | 이번 r1 (새 gateway) | EXP-168 PolyServe r1 / r2 (옛 gateway `05745270`) |
|---|---|---|
| 채점된 도착 수 | 97,999 | 98,012 / 98,044 |
| 모든 도착 | 53.84 | 53.37 / 54.04 |
| admitted | 65.83 | 65.18 / 65.98 |
| 거절률 % | 18.01 | 17.93 / 17.85 |
| token goodput tok/s | 8,666 | 8,637 / 8,669 |
| 엔진이 만든 토큰 전체 tok/s | 11,101 | 11,130 / 11,107 |
| chat 모든 도착 / admitted / 거절률 | 52.18 / 55.15 / 5.24 | 51.30 / 54.21 / 5.22, 52.41 / 55.30 / 5.04 |
| deepresearch | 42.72 / 98.78 / 56.48 | 43.07 / 99.07 / 56.27, 42.85 / 99.27 / 56.53 |
| swe | 80.46 / 99.39 / 18.97 | 80.66 / 99.42 / 18.80, 80.65 / 99.29 / 18.67 |
| preemption 횟수 (세션이 달라 비교하지 않는다) | 0 | 0 / 0 |
| KV 사용률 평균 / p90 | 0.42 / 0.61 | 0.44 / 0.60 |

메모(판정은 r2가 끝난 뒤 규칙 (나)로 한다): 1회차 값은 모든 지표가 EXP-168의 두 run 사이이거나 0.5점 안에 있다. 예측(70B에서는 gateway의 대체 배치가
0건이었으므로 gateway 수정이 PolyServe 값을 바꾸지 않는다)과 같은 방향이다.

run 2(후보 3 r1, `261122_0803_exp175hr1_fsxc3_shift`)는 01:02에 떴다: scheduler 파드 `5d52124e`, 기동 줄에 `--fluidserve-class-instance-cap=false`,
`--fluidserve-kv-price=on`, `verified: every FluidServe flag matches what was applied`.
