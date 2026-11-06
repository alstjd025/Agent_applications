# EXP-149 — 같은 상태에서 horizon만 25로 바꿔 다시 판정하면 어느 항이 어느 클래스의 판정을 바꾸는가 (shadow 판정)

세션 접두어: `exp149h`

작성 2026-11-06, **실행 전**. 사용자와 합의한 분석(2026-11-06 대화): EXP-148 h25 반복 1에서 chat·swe 거절은 줄고(16.2 → 5.8 %, 25.4 → 17.2 %)
deepresearch 거절은 늘었다(30.3 → 37.5 %). 두 run은 서로 다른 요청을 받아 서로 다른 상태에 도달하므로, 게이지를 비교해서는 horizon이 들어가는 세 곳 중
어느 것이 판정을 바꿨는지 알 수 없다. 그래서 상태를 고정하고 h만 바꿔 다시 판정한다.

## horizon이 판정에 들어가는 세 곳 (코드: `fluidserve.go` buildFlux·evaluate·costOf, `fluidserve_capacity.go` maxKvForAllowance)

| 이름 | 무엇 | h를 줄이면 |
|---|---|---|
| proj (투영) | 한 horizon 뒤의 KV = 지금 KV + 처리 중 decode 수 × h − h 안에 풀려날 KV(outflow). 메모리 판정과 토큰당 시간 추정의 KV 항에 들어간다 | 증가량과 outflow가 같이 준다 |
| pace (토큰당 시간 추정) | prefill이 끼는 iteration의 비중 = 대기 prefill 청크 수 ÷ h (meanStepMs, 토큰당 시간 상한에서 나온 KV 한도 capKv), 그리고 도착할 prefill을 예상하는 기간(h × 지금 토큰당 시간) | 대기 prefill이 같아도 비중이 커진다. 도착 예상분은 기간에 비례해 준다 |
| charge (청구) | 들어오는 요청에 미리 잡는 KV = prompt + min(h, 기대 출력) | 출력 쪽이 준다 |

## 구성

| 항목 | 값 |
|---|---|
| 스케줄러 | `783a574e` (`staging/scheduler-shadowh`, 바깥 `83299fb`): `3a1d2354` + `--fluidserve-shadow-horizon` |
| 게이트웨이 | `05745270` (EXP-147·148과 같음) |
| arm | `fsexactprogshadow25` = `fsexactprog`(horizon 100으로 결정) + `--fluidserve-shadow-horizon=25`. **결정은 horizon 100의 것이다** |
| shadow 변형 | 후보 인스턴스 하나마다 같은 flux로 판정을 다섯 번 다시 한다: `same`(전부 100, 검증용), `all`(전부 25), `pace`(pace만 25), `proj`(proj만 25), `charge`(charge만 25) |
| 카운터 | `scheduler_fluidserve_shadow_candidate_total{variant,tier,real,shadow}`, `..._shadow_decision_total{variant,tier,real,shadow}`(인스턴스 상한을 적용한 뒤 남은 후보 중 하나라도 feasible인가), `..._shadow_infeasible_total{variant,reason,tier}`, 히스토그램 `..._shadow_meanafter_delta_ms`, `..._shadow_headroom_delta_tokens` |
| trace·워크로드 | EXP-147·148과 같다 |
| 측정 경로 변경 | `set_scheduler_profiling.py`에 `FS_SHADOW_HORIZON` 대응과 검증 항목 한 줄씩. **EXP-148이 끝난 뒤 체인이 넣는다**(그 전에는 EXP-148이 그 파일을 읽는다). 넣기 전 파일이 패치를 만든 시점과 같은지 md5로 확인한다 |
| 체인 | `/home/nxclab/scratch/exp149/run.sh`, 로그 `/home/nxclab/tools/exp149_chain.log`. EXP-148 체인이 DONE으로 끝나기를 기다린 뒤 시작 |

단위 테스트(`fluidserve_shadow_test.go`): 실제 horizon으로 다시 판정하면 evaluate()의 판정과 같다(feasible·infeasible 둘 다 나오는 상태에서), shadow horizon = 실제 horizon이면
다섯 변형 모두 실제 판정과 같다, pace의 h를 줄이면 prefill이 대기 중인 인스턴스에서 예상 토큰당 시간이 커진다.

## 판정 (실행 전에 정함)

이 run은 결정을 바꾸지 않으므로 점수 비교가 목적이 아니다. 점수는 EXP-147 `fsexactprog`와 지표별 규칙으로 대조만 한다(같은 구성이어야 하므로 차이가 문턱 안이어야
shadow 계산이 결정 경로를 건드리지 않았다고 본다).

1. **유효성**: `same` 변형의 후보 판정 불일치가 전체 후보 평가의 1 % 이하. 넘으면 다른 변형의 수치를 쓰지 않는다(재구현이 evaluate()와 다르다는 뜻).
2. **클래스별 결정 수준 효과**: 클래스(tier 25 swe, 50 chat, 100 deepresearch)마다, 결정 평가 중 "실제 feasible 후보 있음 → shadow `all`에서 없음"과 그 반대의 비율.
   EXP-148의 방향(deepresearch는 h25에서 더 막히고 chat은 덜 막힌다)이 같은 상태에서도 나오는지 본다.
3. **어느 항인가**: 같은 비율을 `pace`, `proj`, `charge` 각각에 대해 낸다. 한 항의 비율이 `all`의 비율의 절반 이상이면 그 항이 그 클래스의 변화를 주로 만든다고 적는다.
   셋의 합이 `all`과 크게 다르면(상호작용) 그렇게 적는다.
4. 같이 본다: 변형별 실패 사유(gate, incumbents, memory, pace_kv)의 클래스별 증감, 예상 토큰당 시간 변화량(`meanafter_delta`)의 클래스별 평균.

한계: 판정은 같은 상태에서 한 번 더 하는 것이지만, h25였다면 그 뒤에 달라졌을 상태(받은 요청이 달라지는 것)는 반영하지 않는다. 결정 평가는 대기(pend) 중 재평가를
포함하므로 요청 수가 아니라 평가 횟수다.
