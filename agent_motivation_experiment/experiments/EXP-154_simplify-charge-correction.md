# EXP-154 — 단순화 A/B: 미리 잡는 KV 청구를 지우고(A), 지연 보정까지 지우면(AC) 결과가 그대로인가

세션 접두어: `exp154h`

작성 2026-11-09, **실행 전**. 사용자 요청(2026-11-08 대화, EXP-153 보고 뒤 "진행해").

## 왜

EXP-153 제거 후보 장부(같은 상태에서 판정만 다시 계산):
- A(새 요청에 min(h, 기대 출력)만큼 미리 잡는 KV 청구를 지우고 prompt만 청구): 결정 평가의 0.06 % 이하에서만 판정이 바뀜 → 지워도 된다.
- C(지연 보정 제거): 1.0~2.5 %에서 양방향으로 바뀜. 보정값 자체는 한 시간 run 내내 fleet 0.97~1.03, 인스턴스별 0.99~1.01(p1~p99, EXP-151 대조 r1).
장부는 같은 순간의 판정만 비교하므로, 실제로 지웠을 때 결과가 같은지 한 시간 trace로 확인한다.

## 구성

| 항목 | 값 |
|---|---|
| 스케줄러 | `359aac2c` (`staging/scheduler-simplify1`, 바깥 `37e4d52`): `--fluidserve-charge-prompt-only`, `--fluidserve-latency-correction` 추가(기본값은 이전과 같음). shadow·장부는 꺼져 있다(결정 시간 증가 없음) |
| 게이트웨이 | `05745270` |
| arm | `fsexactprog`(대조), `fsexactprogA`(prompt만 청구), `fsexactprogAC`(prompt만 청구 + 지연 보정 끔) |
| 순서 | 반복이 바깥 루프: r1 대조 → A → AC, r2 같은 순서. 약 7.5시간 |
| 드라이버·체인 | `k8s/exp07/run_exp154_hour.sh`, `/home/nxclab/scratch/exp154/run.sh`, 로그 `/home/nxclab/tools/exp154_chain.log` |

단위 테스트: prompt만 청구한 판정이 장부 변형 A의 예측과 같다; 보정을 끄면 평균 iteration 시간이 보정 전 모델 값이다.

## 판정 (실행 전에 정함)

지표별 판정(아홉 지표; 문턱 모든 도착 계열 4점·admitted 계열 1점 또는 반복 범위). 같은 세션의 대조와 비교.
- A가 아홉 지표 모두 차이 없음 → A를 코드에서 지운다.
- AC가 아홉 지표 모두 차이 없음 → 지연 보정도 지운다. 하나라도 다르면 지우지 않고 어느 지표인지 적는다.
- 같이 본다: preemption(같은 세션), goodput, deepresearch TTFT 초과, 남는 기록. 마지막 구간의 갈림(EXP-150·153: 엔진 하나가 deepresearch 쪽으로 넘어가 chat 거절 급증)이
  어느 run에서 일어나는지 5분 창 chat 거절로 표시한다 — 갈림이 한 arm의 한 run에만 있으면 그 차이를 arm의 효과로 읽지 않는다.
