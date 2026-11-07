# EXP-151 — horizon은 100으로 두고 토큰당 시간 추정의 prefill 비중 분모만 25·50으로 바꾸면 h25·h50의 결과가 나오는가

세션 접두어: `exp151h`

작성 2026-11-08, **실행 전**. 사용자 요청(2026-11-08 대화, EXP-150 보고 뒤 "그것도 진행해").

## 왜

- EXP-148(한 시간 trace): h25·h50이 h100보다 모든 도착 기준 5.7~6.6점 높았고(chat·swe가 더 받아들여짐), h25는 deepresearch 모든 도착이 7.4점 낮았다.
- EXP-149·150(shadow 판정): 같은 상태에서 horizon을 바꿨을 때 판정이 바뀌는 것은 주로 **토큰당 시간 추정(pace)**에서 나온다 — 대기 prefill 청크 수 ÷ h를
  prefill이 끼는 iteration의 비중으로 쓰는 항(그리고 도착 prefill 예상 기간). h25 방향 막힘의 89~97 %, h200 방향 풀림의 55~65 %. 투영(증가량·outflow)은 그다음이고,
  새 요청에 미리 잡는 KV(charge)는 0이다.
- 그러면 투영과 charge는 100으로 두고 pace의 분모만 바꿔도 EXP-148의 h25·h50 결과가 대부분 나와야 한다. 나오면 "horizon 효과"는 사실상 prefill 비중 계산의 효과이고,
  고칠 곳이 그 식 하나로 좁혀진다.

## 구성

| 항목 | 값 |
|---|---|
| 스케줄러 | `4585b1f4` (`staging/scheduler-pacehz`, 바깥 `c8c831e`): `783a574e` + `--fluidserve-pace-horizon` (0 = horizon을 따름) |
| 게이트웨이 | `05745270` |
| arm | `fsexactprog`(대조, horizon 100), `fsexactprogp25`·`fsexactprogp50`(horizon 100 + pace horizon 25·50). 투영·charge는 100 |
| 순서 | 반복이 바깥 루프: r1 대조 → p25 → p50, r2 같은 순서. run 하나 약 70분, 전체 약 7.5시간 |
| 대조를 같은 세션에 넣는 이유 | EXP-150에서 같은 구성의 엔진 iteration 시간이 세션에 따라 6 % 달랐고 chat 모든 도착이 6.5점 움직였다. EXP-147·148의 값은 참고로만 쓴다 |
| 드라이버·체인 | `k8s/exp07/run_exp151_hour.sh`, `/home/nxclab/scratch/exp151/run.sh`, 로그 `/home/nxclab/tools/exp151_chain.log` |

단위 테스트: pace horizon 25(horizon 100)의 판정이 shadow 판정의 `pace` 변형(25)이 그 상태에서 예측한 판정과 정확히 같다.

## 판정 (실행 전에 정함)

지표별 판정(모든 도착, admitted, 거절률, 클래스별 admitted, 클래스별 모든 도착; 문턱 모든 도착 계열 4점·admitted 계열 1점 또는 두 조건 각각의 반복 범위 중 큰 것).

1. p25·p50 각각을 같은 세션의 대조와 비교한다.
2. **"pace만으로 horizon 효과가 나온다"**: p25(p50)의 대조 대비 차이가, EXP-148의 h25(h50) 대 h100 차이(모든 도착 +6.6(+5.7), chat 모든 도착 +10.7(+8.1),
   swe 모든 도착 +8.7(+9.1), deepresearch 모든 도착 −7.4(−3.75))와 같은 방향이고, 문턱을 넘는 지표가 같으면 그렇게 적는다. 크기가 절반 미만이면 "일부만 나온다".
3. 같이 본다: deepresearch TTFT 초과 건수, goodput, preemption(같은 세션 안에서), 남는 기록(run마다 10건 이하여야 유효).

예상: p25·p50이 EXP-148의 h25·h50과 비슷한 결과(chat·swe 증가, deepresearch 감소). 다만 shadow에서 투영의 몫도 27~43 %였으므로 크기는 h25·h50보다 작을 수 있다.
