# EXP-150 — 같은 상태에서 horizon을 200으로 늘리면 어느 항이 판정을 바꾸는가 (shadow 판정, EXP-149의 반대 방향)

세션 접두어: `exp150h`

작성 2026-11-07, **실행 전**. 사용자 요청(2026-11-06 대화, EXP-149 보고 뒤 "1번도 진행").

## 왜

EXP-148에서 h200은 받아들인 요청이 마감을 지키지 못했다(2회 평균 deepresearch admitted 73.3, swe 90.5, deepresearch TTFT 10 s 초과 run당 약 3,870건,
preemption run당 약 2,450). 받아들일지 정하는 판정이 h200에서 너무 느슨해졌다는 뜻인데, horizon이 들어가는 세 곳(EXP-149 기록의 표: proj 투영, pace 토큰당 시간 추정의
prefill 비중과 도착 prefill 예상 기간, charge 새 요청에 미리 잡는 KV) 중 어느 것이 느슨하게 만드는지는 모른다. EXP-149가 h25에 대해 한 것을 h200에 대해 한다.

## 구성

EXP-149와 같고 shadow horizon만 다르다.

| 항목 | 값 |
|---|---|
| 스케줄러 / 게이트웨이 | `783a574e` / `05745270` |
| arm | `fsexactprogshadow200` = `fsexactprog`(horizon 100으로 결정) + `--fluidserve-shadow-horizon=200` |
| 드라이버 | `k8s/exp07/run_exp150_hour.sh` (EXP-149 드라이버 사본 + 이 arm) |
| 체인 | `/home/nxclab/scratch/exp150/run.sh`, 로그 `/home/nxclab/tools/exp150_chain.log`. **run이 끝나면 정리 전에 스케줄러 파드의 `/metrics`를 `scheduler_metrics_final.txt`로 저장한다**(수집기 파일에 shadow 카운터가 없어서, EXP-149에서는 파드가 살아 있던 덕에 읽었다) |

## 판정 (실행 전에 정함)

EXP-149와 같은 규칙, 방향만 반대다. "풀림" = 실제(h100)로는 feasible 후보가 없었는데 h200에서는 생김.

1. 유효성: `same` 변형의 후보 판정 불일치 1 % 이하.
2. 클래스별 결정 수준: 막힘·풀림 비율(결정 평가 수 대비)을 `all`, `pace`, `proj`, `charge`로 낸다.
3. 한 항의 풀림 비율이 `all`의 절반 이상이면 그 항이 그 클래스를 느슨하게 만드는 주된 원인이라고 적는다.
4. 점수는 EXP-147 `fsexactprog`와 아홉 지표로 대조(결정이 바뀌지 않았는지 확인).

예상: EXP-149의 거울상으로 pace가 주된 원인(prefill 비중이 1/200로 작아져 긴 프롬프트를 싸게 본다). 다만 h200에서는 charge(min(200, 기대 출력))와
proj(outflow 창 200 step, 증가량 n×200)가 h25 때보다 커서 몫이 생길 수 있다.

한계: EXP-149와 같다(같은 순간의 판정만 비교, 평가 횟수 기준, warmup 포함 누적).

## 결과 (2026-11-07 23:49~2026-11-08 00:58, `261107_0650_exp150hr1_fsexactprogshadow200_shift`)

엔진 4/4, 도착률 27.43 req/s, 측정 구간 파드 각 1개, 남는 기록 0건. 카운터는 체인이 정리 전에 저장한 `/home/nxclab/scratch/exp150/scheduler_metrics_final.txt`
(파드 기동 이후 누적, warmup 포함), 분석 `/home/nxclab/scratch/exp149b/shadow.py`.

**1. 유효성**: `same` 변형 후보 판정 불일치 **0건**(swe 106,972 · chat 383,104 · deepresearch 323,188회 평가 중).

**2·3. 결정 수준**("풀림" = 실제 h100으로는 feasible 후보가 없었는데 h200에서는 생김, 결정 평가 수 대비; 막힘은 세 클래스 모두 0.04 % 이하)

| 클래스 | 결정 평가 | 실제 feasible | all 풀림 | pace만 | proj만 | charge만 |
|---|---|---|---|---|---|---|
| swe | 26,743 | 35.8 % | 42.7 % | 27.9 % (all의 65 %) | 17.5 % (41 %) | 0.0 % |
| chat | 95,776 | 52.8 % | 26.5 % | 14.6 % (55 %) | 14.6 % (55 %) | 0.0 % |
| deepresearch | 80,797 | 19.2 % | **54.4 %** | 33.6 % (62 %) | 20.9 % (38 %) | 0.0 % |

- 같은 상태에서 h를 200으로 늘리면 세 클래스 모두 크게 느슨해진다. deepresearch는 결정 평가의 54 %에서 받을 수 있는 인스턴스가 새로 생긴다(실제 feasible 19 %).
- 판정 규칙(한 항이 all의 절반 이상): **pace는 세 클래스 모두 주된 원인**, **proj는 chat에서도 주된 원인**(55 %), swe·deepresearch에서는 38~41 %. charge는 0.
  pace와 proj의 합이 all과 거의 같다(swe 45.4 대 42.7, chat 29.2 대 26.5, deepresearch 54.5 대 54.4) — 두 항이 대체로 따로 더해진다.
- 항별로 무엇이 풀리나(후보 평가 대비 실패 비율, 실제 → 변형):
  - proj: 물리 KV 판정(memory) 2.4~5.1 % → **0 %**, headroom +5.4~7.9만 토큰. outflow 창이 200 step이 되어 그 안에 끝날 요청이 많아지므로 풀려날 KV를 많게 잡는다.
  - pace: 처리 중인 요청 보호(incumbents) deepresearch 29.4 → 15.1 %, swe 19.4 → 9.8 %, 예상 토큰당 시간 −0.5~−1.8 ms. prefill 비중(대기 청크 ÷ h)이 작아져 긴 프롬프트를 싸게 본다.
- **EXP-149(h25)와 비교**: h25에서는 pace가 막힘의 89~97 %를 만들고 proj의 몫이 작았다. h200에서는 pace가 여전히 가장 크지만 proj가 38~55 %로 커진다.
  outflow는 h에 비례하는 것보다 빠르게 커지는데(창이 길어지면 그 안에 끝나는 요청의 비율이 커진다), 짧은 쪽에서는 이미 작아서 줄어들 여지가 적다.
- EXP-148 h200의 과대 admission(deepresearch admitted 73.3, TTFT 초과 run당 약 3,870건)과 같은 방향: deepresearch가 가장 많이 풀리고, 그 몫은 pace(긴 프롬프트를 싸게 봄)와
  proj(풀려날 KV를 많게 봄)가 나눠 가진다.

**4. 점수**: EXP-147 `fsexactprog` 2회와 비교하면 **chat 모든 도착 −6.47(문턱 4)로 다르다**. 나머지 여덟 지표는 차이 없음(모든 도착 −3.72, 거절률 +3.67로 문턱 직전).
EXP-149(같은 바이너리, shadow 25)는 EXP-147과 0.3점 안이었다.
- shadow가 결정을 바꿨을 가능성: shadow가 부르는 함수(expectedOutflow, completionProb·survivalAt, meanStepMs, maxKvForAllowance, prefillStepMs, carriedPrefillTokens)는
  모두 값을 읽기만 하고 상태를 바꾸지 않는다(코드로 확인). shadow horizon 값만 다른 같은 코드가 EXP-149에서는 점수를 바꾸지 않았다.
- 세션 차이: 이 run에서 엔진 iteration 시간 중앙값이 **48.9 ms**로 EXP-147·149(46.0 ms)보다 6 % 길다(엔진당 처리 중 요청 수는 148로 같음). 스케줄러가 그 느린 엔진에 맞춰
  chat(50 ms 예산)을 더 거절했다(chat 거절 22.6 % 대 16 %). GPU 전력 제한은 1000 W 그대로였고(powertest 로그), 엔진이 왜 느렸는지는 확인하지 못했다.
  preemption 1,011(세션 비교 안 함).
- shadow 결과(같은 상태에서의 판정 비교)는 그 상태가 무엇이든 유효하지만, 상태 분포가 EXP-149와 다르다는 점은 같이 적는다.
