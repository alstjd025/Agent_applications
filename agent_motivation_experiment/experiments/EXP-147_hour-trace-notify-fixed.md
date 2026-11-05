# EXP-147 — 완료 통지 수정이 모두 들어간 상태의 한 시간 trace: 엔진 대기 요청 제외만 켠 구성과 첫 토큰 마감까지 켠 구성

세션 접두어: `exp147h`

작성 2026-11-05, **실행 전**. 사용자 요청: `fsexactprog`와 `fsexactprogdl`을 한 시간 trace로 2회씩, arm 순서를 번갈아 돌리고
절반(반복 1이 끝난 시점)에 전체 결과를 한 번 본다.

## 왜

- EXP-143(2026-10-20)의 한 시간 trace `fsexactprog`(모든 도착 78.6, deepresearch admitted 95.6)는 완료 통지가 빠지는 두 경로가 고쳐지기 전의
  바이너리(스케줄러 `3251836a`, 게이트웨이 `590242b0`)로 돌았다. 그 run에서는 통지를 받지 못한 기록이 20분 나이 제한까지 목록에 남았다.
- 통지 경로 수정(EXP-146: 통지를 배치한 스케줄러 파드로 직접 보냄 `c0fb7ab`, keepalive 재연결로 게이트웨이 ID가 바뀌어도 통지를 보냄 `5dec8d6`,
  러너가 측정 전에 종료 중인 이전 파드를 지움 `42296c0`)이 모두 들어간 상태로 한 시간 trace를 측정한 적이 없다.
- 첫 토큰 마감을 feasibility에 넣는 조건(`--fluidserve-deadline-feasible=true`, 논문 구성에서는 꺼짐, revision.md R9)은 정적 표준 혼합(EXP-142)에서만
  확인했다(deepresearch 100.0, 모든 도착 75.6). EXP-143의 deepresearch 실패 623건은 15~25분에 한 인스턴스로 deepresearch가 몰려 TTFT가 10 s를
  넘은 것이었고, 이 조건이 막았을 모양이다.

## 구성

| 항목 | 값 |
|---|---|
| 스케줄러 | `3a1d2354` (`staging/scheduler-notifyfix`, 바깥 `c0fb7ab`: 파드 주소 헤더, UUID 대기 ID 처리 `41a667f`, 읽기 전용 `/debug/fluidserve-registry`) |
| 게이트웨이 | `05745270` (`staging/gateway-notifyfix2`, 바깥 `5dec8d6`: 파드로 직접 통지 + ID 비교 제거). **처음 배포하는 바이너리** |
| 드라이버 | `k8s/exp07/run_exp147_hour.sh` — EXP-143 드라이버 사본. 바뀐 것: `fsexactprogdl` arm, 그리고 바이너리 확인이 마지막 파드 하나가 아니라 **모든** 스케줄러·게이트웨이 파드를 본다(교체 중인 파드도 phase가 Running이라 잘못된 파드가 답할 수 있었다) |
| trace·워크로드 | EXP-143과 같다: `dyn60_shift_m2Am1B_b1045.csv`, `mix_dyn60_shift_m2Am1B_b1045_t75.json` |
| arm | `fsexactprog`(notified residency + 엔진 대기 요청 제외), `fsexactprogdl`(여기에 첫 토큰 마감 조건) |
| 순서 | r1 `fsexactprog` → r1 `fsexactprogdl` → r2 `fsexactprog` → r2 `fsexactprogdl` (반복이 바깥 루프). arm 하나 약 75~80분, 전체 약 5시간 20분 |
| 체인 | `/home/nxclab/scratch/exp147/run.sh`, 로그 `/home/nxclab/tools/exp147_chain.log` |

### 이전 파드가 남는 문제에 대한 대비

1. 조건마다 러너의 `restart_llumnix`가 스케줄러·게이트웨이를 다시 띄운 뒤 종료 중인 이전 파드를 지우고 사라진 것을 확인한 다음 warmup을 시작한다.
   `set_arm`의 정책 전환(재시작 한 번 더)으로 생긴 이전 파드도 이 단계에서 지워진다.
2. 그래도 측정 중에 교체가 일어나면, 스케줄러가 응답 헤더로 알려 준 파드 주소로 통지가 가고(③), 재연결로 ID가 바뀌어도 통지를 보낸다(④).
3. 체인이 60초마다 스케줄러·게이트웨이 파드 목록(이름, 종료 표시, 재시작 수)을 `pods.log`에, 30초마다 모든 스케줄러 파드의 목록 snapshot을 `snap/`에 적는다.
   스케줄러나 게이트웨이 파드가 10분 연속 둘 이상이면 체인 로그에 `WARN` 줄을 쓴다. 로그를 계속 따라 읽지 않는다(`kubectl logs -f`가 게이트웨이를 느리게 만든 적이 있다).

## 판정 (실행 전에 정함)

ladder95, swe는 TTFT 7 s + 75 ms/token. 한 시간 trace의 같은 arm 반복 간 편차가 3.8점(EXP-109 계열)이므로:

- **남는 기록**: snapshot 시각에 목록에 있는데 클라이언트 기록상 마지막 토큰을 받은 지 2초가 넘은 요청. 측정 내내 run마다 **최대 10건 이하**이면
  "통지 수정이 한 시간 동안 유지된다". 넘으면 그 요청들이 어느 파드에서 언제 스케줄링되었는지부터 본다.
- **두 arm 비교**(2회 평균): 모든 도착 기준 차이가 max(4점, 두 arm의 반복 범위)를 넘을 때만 "다르다".
- **deepresearch**: `fsexactprogdl`의 deepresearch admitted가 2회 모두 98 이상이고 `fsexactprog`가 98 미만이면, 한 시간 trace에서도 deepresearch 실패의 원인은
  첫 토큰 마감이 feasibility에 없는 것이다(EXP-142 정적 결과와 같은 방향).
- 같이 본다: 클래스별 admitted·거절률, token goodput, deepresearch TTFT 초과 건수와 그것이 몰린 인스턴스·시간 구간, preemption(이 세션 안에서만 비교).
- 참고 값(다른 세션이라 판정에는 쓰지 않는다): EXP-143 `fsexactprog` 78.6, 논문 구성 `fsv3capgnofrct75` 76.5.

예상: 남는 기록은 0 근처. `fsexactprogdl`이 deepresearch를 98 이상으로 올리고, 모든 도착은 두 arm이 구별되지 않는다.

## 중간 결과: 반복 1 (2026-11-05 12:02~14:15, `exp147hr1`)

ladder95, 대시보드 채점기. 두 run 모두 엔진 4/4, 도착률 27.43 req/s(EXP-143과 같음). 측정 구간 내내 스케줄러·게이트웨이 파드 각 1개(`pods.log`).
"모든 도착" = 거절을 위반으로 세고 미완료는 뺀 값. preemption은 엔진 4대 합. EXP-143 두 줄은 다른 세션(2026-10-20)이라 참고 값이다.

| run | 모든 도착 | admitted | 거절 % | goodput tok/s | chat adm / 거절 | deepresearch adm / 거절 | swe adm / 거절 | deepresearch 실패 (TTFT > 10 s) | preemption |
|---|---|---|---|---|---|---|---|---|---|
| r1 `fsexactprog` | 79.3 | 99.97 | 20.4 | 11,629 | 100.0 / 16.2 | **100.0** / 30.3 | 99.8 / 25.4 | 3 (3) | 650 |
| r1 `fsexactprogdl` | 78.7 | 99.99 | 21.1 | 11,562 | 100.0 / 17.1 | 100.0 / 29.4 | 99.9 / 27.5 | 1 (1) | 618 |
| 참고: EXP-143 `fsexactprog` (수정 전 바이너리) | 78.4 | 99.11 | 20.7 | 11,513 | 100.0 / 15.7 | 95.6 / 31.1 | 99.4 / 29.1 | 639 (619) | 1,074 |
| 참고: EXP-143 논문 구성 `fsv3capgnofrct75` | 76.3 | 99.96 | 23.5 | 11,348 | 100.0 / 21.8 | 100.0 / 27.8 | 99.7 / 25.1 | 1 (1) | 767 |

- **남는 기록 0건**: 30초 snapshot 117·118개 모두에서, 목록에 있는데 클라이언트가 마지막 토큰을 받은 지 2초가 넘은 요청이 없었다. 목록 크기는 중앙값 690·664건이고,
  확인한 snapshot 하나에서 목록 779건 전부가 클라이언트 기록에 있었으며 그 시각 클라이언트 기준 처리 중 요청은 781건이었다(ID 대응이 동작한다는 확인).
  판정 "최대 10건 이하" 충족 — 한 시간 동안 통지 수정이 유지된다(반복 1 기준).
- 반복 1에서는 **`fsexactprog`만으로도 deepresearch admitted가 100.0**이다. EXP-143의 같은 arm(95.6, 15~25분에 한 인스턴스에서 TTFT 초과 619건)과 다르다.
  두 run의 차이는 바이너리(통지 수정 둘, UUID 대기 ID 처리)와 세션이다. 반복 2까지 보고 판정한다.
- 두 arm의 모든 도착 차이 0.6점 — 판정 문턱 4점보다 훨씬 작다(반복 1만으로는 판정하지 않음).
