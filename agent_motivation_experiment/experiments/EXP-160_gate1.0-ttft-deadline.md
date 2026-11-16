# EXP-160: 판정선 1.0 + 첫 토큰 마감 판정 (v0.6-후보4)

**날짜**: 2026-11-16 (실행 전 작성)
**질문**: v0.6-후보3(판정선 1.0)에서 swe가 놓친 첫 토큰 마감을, 꺼져 있던 첫 토큰 마감 판정(`--fluidserve-deadline-feasible`)을 켜서 막으면,
판정선 1.0의 이득(모든 도착 +5.0)을 지키면서 나빠진 두 지표(swe admitted −3.5, deepresearch 모든 도착 −5.2)를 되돌릴 수 있는가.

## 배경

- EXP-159: 판정선 1.0은 모든 도착 77.95 → 82.93, 거절률 21.8 → 16.4 %, chat·swe 모든 도착 +7.3·+10.3, 분리 유지. 그러나 swe admitted 99.65 → 96.15, deepresearch 모든 도착 70.0 → 64.8로 채택하지 않았다.
  받은 swe의 마감 위반 400~473건 중 절반(190~215건)이 첫 토큰 초과(TTFT > 7 s, 중앙값 8.0~8.1 s, 최대 14.4 s), 절반이 평균 토큰 간격 초과다.
- v0.5(와 후보3)는 가능/불가 판정에서 첫 토큰 시점을 보지 않는다(`deadlinefeasible=false`): 토큰당 시간과 KV만 통과하면 그 인스턴스의 prefill 대기열이 길어도 보낸다.
  첫 토큰은 pend할지 거절할지를 정할 때만 본다. EXP-147에서 v0.5에 이 판정을 켰을 때(`fsexactprogdl`)는 아홉 지표 모두 차이 없었다 — v0.5에서는 첫 토큰을 넘기는 요청이 드물었다.
- 사전 확인(`analysis_scripts/exp160/ttft_check.py`, EXP-159 결정 로그의 route 줄 `waited + prefillest`와 채점 결과를 맞춤): 판정선 1.0 run에서 TTFT를 넘긴 swe 228·208건 중 **219·199건(약 96 %)은 이 판정을 켰다면 route된 그 순간 그 인스턴스가 불가**였다.
  대가로 TTFT 안이었던 swe 940·1,040건, chat 3,758·3,429건도 그 순간 불가였을 것이다(예측이 비관적: TTFT를 넘긴 swe에서도 예측이 실제보다 1.3~1.5 s 길다). 불가는 그 인스턴스만 빠지는 것이라 다른 인스턴스로 가거나 pend 후 재평가된다.

## 구성

| 항목 | 값 |
|---|---|
| arm | `fsexactprog`(v0.5), `fsg10dl`(v0.6-후보4 = v0.5 + `FS_GATE_UTIL=1.0` + `FS_DEADLINE_FEASIBLE=true`) |
| 첫 토큰 마감 판정 | 기다린 시간 + 예측 prefill 시간(자기 prefill + 앞에 쌓인 prefill) > TTFT budget이면 그 인스턴스는 불가. 배치 후 지연 상한은 더하지 않는다(`deadlinedelay=false`) |
| 바이너리 | 스케줄러 `7dc2355c`(v0.5.8), 게이트웨이 `05745270` — 코드 변경 없음, 기존 플래그 둘 |
| trace | 한 시간 `dyn60_shift_m2Am1B_b1045`, h100 |
| 반복 | 2회, 순서를 번갈아: r1 v0.5 → 후보4, r2 후보4 → v0.5. 약 5시간 |
| 드라이버·체인 | `k8s/exp07/run_exp160_hour.sh`, `/home/nxclab/scratch/exp160/run.sh`, 로그 `/home/nxclab/tools/exp160_chain.log`. SparkGen lease가 있으면 놓을 때까지 기다린다 |

## ⚠ 정정 (2026-11-16): 이 실험의 FluidServe 후보 arm은 swe TTFT budget이 다른 워크로드 설정으로 돌았다

드라이버(`run_exp15x/16x_hour.sh`의 `run_cell`)는 arm 이름이 목록(`case "$arm" in fsv3capgnofrct75|fsexact|fsexactprog|...|fsc2h200)`)에 있을 때만
swe TTFT 7,000 ms 설정 `mix_dyn60_shift_m2Am1B_b1045_t75.json`을 쓰고, 목록에 없으면 경고 없이 원래 설정 `mix_dyn60_shift_m2Am1B_b1045.json`(swe `ttft_ms` **11,800**)으로 간다.
이 실험에서 새로 만든 arm(EXP-158 `fsshadowv`, EXP-159 `fsg10`, EXP-160 `fsg10dl`)을 그 목록에 넣지 않았다. 각 run의 `run_config.json`의 `workload_config.slo.swe.ttft_ms`로 확인했다:
v0.5(`fsexactprog`) run은 모두 7000, 위 세 arm의 run은 모두 11800. EXP-147~157의 arm은 모두 7000이다.

스케줄러는 swe의 TTFT budget을 요청에 실려 온 이 값으로 쓴다(토큰당 budget만 `FS_SWE_TBT_MS=75`로 따로 맞춘다). 그래서 이 arm들에서 스케줄러는 swe의 첫 토큰 마감을 11.8 s로 보고
pend 여부(`canWait`)와 첫 토큰 마감 판정(`missesTtftDeadline`)을 했고, 채점은 7 s로 했다. **v0.5 arm과 후보 arm은 판정선 하나가 아니라 swe의 첫 토큰 budget까지 달랐으므로, 이 실험의 arm 간 비교는 판정선(또는 첫 토큰 판정)의 효과로 읽을 수 없다.**
특히 swe(admitted, 모든 도착, 첫 토큰 초과)와 그 영향을 받는 deepresearch 모든 도착의 차이는 이 설정 차이로 생겼을 수 있다. 위의 판정은 무효로 하고 올바른 설정으로 다시 잰다(EXP-161).

EXP-160에 대해 더: 후보4의 첫 토큰 판정이 swe의 첫 토큰 초과를 기대(약 96 %)만큼 막지 못한 이유가 이것이다 — 늦은 swe는 route 순간 (기다린 시간 + 예측 prefill)이 중앙값 9.4~9.6 s로 7 s를 넘었지만 11.8 s 안이어서 판정을 통과했다(route된 swe의 최댓값 11,704 ms).

## 판정 (실행 전에 정함; EXP-159와 같은 규칙)

1. 아홉 지표를 지표마다 따로: "평균 차이 > max(문턱, 두 조건 각각의 반복 범위)"이면 그 지표만 "다르다". 문턱은 모든 도착·거절률·클래스별 모든 도착 4점, admitted·클래스별 admitted 1점.
2. 분리 유지 조건: 후보4의 chat 처리 중 인스턴스 평균이 같은 세션 v0.5 + 0.5대 이하, chat이 3대 이상인 시간이 v0.5 + 15점 이하.
3. 채택: "다르다(나쁨)"가 없고, 모든 도착·거절률·클래스별 모든 도착 중 하나 이상 "다르다(좋음)", 분리 유지 조건 성립. 모두 "차이 없음"이면 v0.5 유지.
4. 같이 적는 것: 받은 swe의 마감 위반 종류(첫 토큰 / 평균 토큰 간격 / 몰림)를 EXP-159 후보3과 비교, swe TTFT p99, preemption, token goodput, chat 거절률(첫 토큰 판정이 chat을 불필요하게 막는지), 마지막 구간 갈림 여부.

**예상**: swe의 첫 토큰 초과는 EXP-159 후보3의 190~215건에서 수십 건 이하로 준다. 평균 토큰 간격 초과(191~226건)는 남으므로 swe admitted는 98 안팎일 것이고, 그러면 swe admitted 지표(문턱 1점)는 여전히 "다르다(나쁨)"일 수 있다.
첫 토큰 판정이 swe를 조금 덜 받게 해 deepresearch 모든 도착은 일부 회복할 수 있다. chat은 판정이 비관적이어서 받는 양이 조금 줄 수 있다.

## 결과

(실행 후 적는다)
