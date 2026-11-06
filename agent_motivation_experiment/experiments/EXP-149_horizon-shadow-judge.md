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

## 실행 기록

- **첫 시도(2026-11-06 22:33~22:41, 측정 없음)**: 체인이 EXP-148 종료를 기다린 뒤 바이너리를 바꾸고 재시작했는데, 드라이버의 `check_stack`이 **종료 중인 이전 스케줄러 파드**
  (`3a1d2354`, EXP-148의 것)까지 검사해서 `ABORT: scheduler ... runs 3a1d2354, want 783a574e`로 멈췄다. 결과 디렉토리·러너 Job은 만들어지지 않았다.
  EXP-147에서 "모든 Running 파드"를 보도록 바꾼 검사가, 체인 자신의 재시작으로 생긴 이전 파드를 잘못 잡은 것이다(EXP-147·148에서는 이전 파드가 우연히 같은 바이너리였다).
  체인을 정리 단계 전에 멈췄다(바이너리·게이트웨이·엔진은 EXP-149 상태 그대로).
- 고침: `run_exp149_hour.sh`의 `check_stack`은 삭제 표시(deletionTimestamp)가 있는 파드를 건너뛴다(러너가 warmup 전에 그 파드를 지운다).
- **두 번째 시도(22:45~)**: `/home/nxclab/scratch/exp149b/run.sh`, 로그 `/home/nxclab/tools/exp149b_chain.log`. `FS_SHADOW_HORIZON` 대응은 첫 시도가 이미 넣었으므로 확인만 한다.
  게이트웨이 replicas는 끝나면 0으로 되돌린다.

## 결과 (2026-11-06 22:48~23:57, `261106_0549_exp149hr1_fsexactprogshadow25_shift`)

엔진 4/4, 도착률 27.43 req/s, 측정 구간 스케줄러·게이트웨이 파드 각 1개, 남는 기록 0건.

**카운터를 어디서 읽었나**: run 디렉토리의 `scheduler.jsonl`은 수집기가 정해 둔 메트릭만 적어서 shadow 카운터가 없다. 체인이 끝난 뒤에도 그 run의 스케줄러 파드
(`scheduler-67d77976dd-4f95v`, 파드 안 `md5sum /proc/1/exe` = `783a574e`)가 살아 있어서 그 파드의 `/metrics`를 그대로 저장했다
(`/home/nxclab/scratch/exp149b/scheduler_metrics_final.txt`, 분석 `/home/nxclab/scratch/exp149b/shadow.py`). 파드 기동 이후 누적값이므로 warmup(부하 낮음)의 결정이 섞여 있다.

**점수(결정이 horizon 100 그대로인지 확인)**: EXP-147 `fsexactprog` 2회와 아홉 지표 모두 차이 없음. 모든 도착 79.36 대 79.39, 가장 큰 차이는 deepresearch 모든 도착 −0.33.
shadow 계산이 결정을 바꾸지 않았다.

**1. 유효성**: `same` 변형(전부 100으로 다시 판정)의 후보 판정 불일치 **0건**(swe 110,044 · chat 364,784 · deepresearch 344,948회 평가 중). 판정 기준 1 % 이하 충족.

**2·3. 결정 수준**: 결정 평가 하나마다 "인스턴스 상한을 적용한 뒤 남은 후보 중 feasible이 있는가"를 실제(h100)와 shadow(h25)로 비교. 비율은 그 클래스의 결정 평가 수 대비
(대기 중 재평가 포함).

| 클래스 | 결정 평가 | 실제 feasible | all(셋 다 25) 막힘 / 풀림 | pace만 | proj만 | charge만 |
|---|---|---|---|---|---|---|
| swe | 27,511 | 35.0 % | 24.4 % / 0.0 % | 21.7 % / 0.5 % | 9.6 % / 0.0 % | 0.0 % / 0.0 % |
| chat | 91,196 | 60.0 % | 27.4 % / 0.1 % | 24.3 % / 2.9 % | 7.1 % / 0.0 % | 0.0 % / 0.0 % |
| deepresearch | 86,237 | 17.2 % | 14.2 % / 0.0 % | 13.7 % / 0.0 % | 6.1 % / 0.0 % | 0.0 % / 0.0 % |

("막힘" = 실제로는 feasible 후보가 있었는데 h25에서는 없음, "풀림" = 반대)

- **같은 상태에서 h를 25로 줄이면 세 클래스 모두 더 막힌다.** 풀리는 경우는 거의 없다(chat pace 2.9 %가 가장 크다).
- **원인 항은 pace(prefill 비중 = 대기 prefill 청크 수 ÷ h)다.** 세 클래스 모두 pace만 바꿔도 all의 막힘의 89~97 %가 나온다(판정 규칙 "all의 절반 이상" 충족).
  proj(증가량·outflow)는 all의 27~43 %(둘이 겹친다), charge(새 요청에 미리 잡는 KV)는 0 — 무시해도 된다.
- 예상 토큰당 시간(후보 평균, shadow − 실제): pace만 바꿔 swe +15.1 ms, chat +8.0 ms, deepresearch +18.1 ms. proj는 +0.4~0.6 ms, charge 0.
  headroom(KV 한도 − 들어간 뒤 KV): pace −50~−67만 토큰(capKv가 준다), proj −3~−5만 토큰(outflow가 준다).
- 후보 수준 실패 사유(후보 평가 대비, 실제 → all): deepresearch incumbents 31 → 85 %, swe incumbents 21 → 66 %, chat incumbents 3 → 26 %;
  memory(물리 KV)는 모든 클래스 1 → 19~25 %인데 이것은 proj가 만든다(proj만 바꿔도 같은 값).

**EXP-148과 어떻게 맞나**: 같은 상태에서 h25는 모든 클래스에 더 엄격하다. 그런데 EXP-148의 h25 run은 chat·swe를 **더 많이** 받았다. 둘은 모순이 아니다 —
h25는 프롬프트가 긴 요청이 있는 인스턴스의 예상 토큰당 시간을 크게 올리므로(deepresearch +18 ms, chat +8 ms), deepresearch를 더 거절하고(EXP-148: 30 → 38 %)
처리 중인 요청을 보호하는 판정(incumbents)이 deepresearch에 가장 많이 걸린다. 그렇게 deepresearch가 덜 들어간 인스턴스는 대기 prefill과 KV가 적은 상태가 되고,
그 상태에서는 chat·swe가 통과한다(EXP-148 h25: preemption 0). 이 마지막 단계(상태가 바뀌어 chat이 더 들어가는 것)는 이 실험이 직접 잰 것이 아니라 두 실험을 합친 해석이다.
한계: shadow는 같은 순간의 판정만 비교하므로 h25였다면 도달했을 상태는 반영하지 않는다.
