# EXP-162: swe가 많은 한 시간 trace (swe 5 : deepresearch 3 : chat 2) — v0.5 대 v0.6-후보4

**날짜**: 2026-11-17 (실행 전 작성)
**질문**: 지금까지의 한 시간 trace는 chat이 도착의 약 2/3이다. swe가 절반인 상황에서 v0.6-후보4(판정선 1.0 + 첫 토큰 마감 판정)가 v0.5보다 받은 요청의 마감을 잘 지키고 더 받는가.
사용자 기준(2026-11-17): **거절이 많은 것은 괜찮고, 받은 요청을 다 지키는 것이 중요하다.** 도착률은 이 run을 보고 조절한다.

## 구성

| 항목 | 값 |
|---|---|
| trace | `traces/dynamic/canonical/dyn60_mix532_b1045.csv` — 지금 trace(`dyn60_shift_m2Am1B_b1045.csv`)와 **도착 시각이 같다**(arrival_s md5 `5256bef0…` 일치), 클래스만 swe 5 : deepresearch 3 : chat 2로 다시 붙였다(10개 블록마다 정확한 비율, 시드 20261117, `traces/dynamic/relabel_fixed_mix.py`): swe 49,621 / deepresearch 29,773 / chat 19,848 / 99,242 |
| 워크로드 설정 | `workload_configs/mix_dyn60_mix532_b1045_t75r.json` — t75 설정과 같고(swe TTFT 7 s) 두 가지만 다르다: 클래스 계획 파일, `swe_config.replay_marker = true` |
| swe 반복 번호 | swe 기록 1,500개를 그대로 순환하면 50 % 비중에서 같은 기록이 평균 약 110 s(가장 바쁜 분 약 67 s)마다 다시 와 prefix 캐시에 통째로 적중한다. 그래서 시스템 프롬프트 뒤 첫 사용자 메시지 앞에 `[replay NN]`을 붙인다(`workloads/codingagent_request_level_poisson/workload.py`, 기본 꺼짐 — 이전 run은 영향 없음). 공통 시스템 프롬프트(약 22,000자)는 그대로 공유된다 |
| arm | `fsexactprog`(v0.5), `fsg10dl`(v0.6-후보4 = v0.5 + `FS_GATE_UTIL=1.0` + `FS_DEADLINE_FEASIBLE=true`) |
| 바이너리 | 스케줄러 `7dc2355c`(v0.5.8), 게이트웨이 `05745270` |
| 반복 | 2회, 순서를 번갈아(r1 v0.5 → 후보4, r2 후보4 → v0.5). 약 5시간 |
| 드라이버·체인 | `k8s/exp07/run_exp162_hour.sh`(변형 `swe532`; `run_cell`의 t75 표는 이제 `shift` 변형에만 적용되고, 설정 검사는 변형마다 맞는 파일을 요구), `/home/nxclab/scratch/exp162/run.sh`(run마다 `run_config.json`에서 클래스 계획 파일·swe `ttft_ms` 7000·`replay_marker` true를 확인해 아니면 WARN), 로그 `/home/nxclab/tools/exp162_chain.log` |

**부하**: 요청당 평균 입력 토큰이 약 1,600(지금 trace) → 약 4,760(swe 6,812 × 0.5 + deepresearch 4,055 × 0.3 + chat 674 × 0.2)으로 약 3배다. 거절이 많을 것이고 그것은 괜찮다.
첫 run 뒤에 확인할 것: 클라이언트 오류 종류(`error_msg`)와 분당 시작된 요청 수(trace가 주는 수를 넘으면 클라이언트가 망가진 것, `exp-run` 스킬 §4e), 스케줄러 결정 지연, swe prefix 적중률(반복 번호가 효과가 있는지).

## 판정 (실행 전에 정함)

1. 아홉 지표(모든 도착, admitted, 거절률, 클래스별 admitted·모든 도착)를 지표마다: "평균 차이 > max(문턱, 두 조건 각각의 반복 범위)"이면 "다르다". 문턱은 모든 도착·거절률·클래스별 모든 도착 4점, admitted·클래스별 admitted 1점.
2. **이번 실험의 주 관심은 admitted(전체·클래스별)**: 사용자 기준대로 받은 요청의 마감 달성이 우선이다. 후보4가 admitted·클래스별 admitted 어느 것에서도 "다르다(나쁨)"가 아니고, 하나 이상에서 "다르다(좋음)"이거나 모든 도착·클래스별 모든 도착 중 하나 이상에서 "다르다(좋음)"이면 이 trace에서 후보4가 낫다고 적는다.
3. 같이 적는 것: 클래스별 받은 요청의 첫 토큰 초과 수와 TTFT p99, 받은 요청의 마감 위반 종류(첫 토큰 / 평균 토큰 간격 / 몰림), preemption, goodput, 분리 지표(클래스별 처리 인스턴스 수), 결정 지연, swe prefix 적중률.
4. 판정 전에 네 run의 `workload_config`가 같은지 대조한다.

**예상**: v0.5는 swe·deepresearch 인스턴스에서 0.9 판정선과 h에 나눈 prefill 간섭 지연으로 긴 프롬프트를 많이 거절해 admitted는 높고 모든 도착은 낮을 것이다.
후보4는 판정선 1.0으로 더 받지만, 첫 토큰 마감 판정이 swe의 첫 토큰 초과를 막아 admitted는 v0.5와 비슷하게 유지될 것이다. swe 평균 토큰 간격 초과(판정선이 swe budget 75 ms에 붙는 것)는 chat이 적은 이 trace에서 더 크게 나타날 수 있다.

## 결과

(실행 후 적는다)
