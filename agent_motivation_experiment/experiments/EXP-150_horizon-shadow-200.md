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
