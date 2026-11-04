# EXP-146 — 완료 통지 수정이 남는 기록을 없애는가, 그리고 deepresearch·swe만 있을 때 admitted가 유지되는가

세션 접두어: `exp146fix`, `exp146dsr`

작성 2026-11-04, **실행 전**. EXP-145에서 찾은 원인(스케줄러 교체 중 완료 통지가 이전 스케줄러 파드로 감)의 수정 확인과, 사용자 질문
"deepresearch·swe만으로 실험해도 chat처럼 admitted가 거의 100%가 되나"의 측정.

## 수정 (둘 다)
- 시스템(바깥 `revision_scheduler` `c0fb7ab`): 스케줄러가 `/schedule` 응답 헤더 `X-Llumnix-Scheduler-Endpoint`에 자기 파드 주소를 넣고, 게이트웨이는 그 요청의
  `/complete`를 그 주소로 먼저 보낸다(실패하면 Service 주소). 스케줄러 `3a1d2354`, 게이트웨이 `f674aa25`(`/home/nxclab/tools/staging/*-notifyfix`).
  스케줄러에는 EXP-145의 읽기 전용 `/debug/fluidserve-registry`와 UUID 대기 ID 처리 수정(`41a667f`)이 같이 들어 있다.
- 운영(안쪽 `42296c0`): 러너의 `restart_llumnix`가 스케줄러·게이트웨이를 다시 띄운 뒤 종료 중인 이전 파드를 제거하고 사라진 것을 확인한 다음 warmup을 시작한다
  (두 deployment 모두 종료 유예 300 s, 두 프로세스 모두 첫 SIGTERM을 무시해 이전 파드가 최대 5분 남았다).

## A. 수정 확인 (`exp146fixr1`)
`fsexactprog`(notified residency + 엔진 대기 요청 제외) 35 req/s 6분, 표준 혼합 `mix_short_m1_t75.json`, 드라이버 `run_exp142_progress.sh`. EXP-145 19절과 같이
t0 + 45 s에 스케줄러와 게이트웨이를 동시에 `rollout restart`(측정 중 교체라 러너의 운영 수정은 이 교체를 막지 않는다 — 시스템 수정만 시험).
5초마다 모든 스케줄러 파드의 snapshot.
- 판정: 새 스케줄러의 남는 기록이 측정 내내 10건 이하면 **시스템 수정이 동작**. 러너 로그에 `old pods removed`가 찍히고 측정 시작 시 스케줄러·게이트웨이 파드가 하나씩이면
  **운영 수정이 동작**. (EXP-145 19절, 수정 전 같은 조건: 16건, 모두 이전 파드가 통지를 받음.)

## B. deepresearch·swe만 (`exp146dsr`)
- 워크로드 `workload_configs/mix_short_drswe_t75.json`(새 파일: `mix_short_m1_t75.json`에서 chat 가중치만 0, deepresearch : swe = 2 : 1), 드라이버
  `k8s/exp07/run_exp146_drswe.sh`(EXP-142 드라이버 사본, 그 mix만 바뀜). 바이너리는 A와 같다.
- arm: FluidServe 논문 구성(`fsv3capgnofrct75`, inferred residency), 고친 notified residency + 엔진 대기 요청 제외(`fsexactprog`).
- 도착률 8·12·16 req/s(480·720·960 rpm), 각 8분, 반복 2회(반복이 바깥 루프). 표준 혼합에서 deepresearch·swe 합이 약 8 req/s다.
- 판정(ladder95, 2회 평균): 각 도착률에서 deepresearch·swe admitted가 **99% 이상**이면 "chat처럼 유지된다". 두 arm의 모든 도착 기준 달성률 차이는
  max(2점, 두 arm 반복 범위)를 넘을 때만 "다르다". 같이 본다: preemption(같은 세션 안에서만), deepresearch TTFT 위반 수, 거절률, outflow 예측/실제(대시보드 기록).
- 예상: 남는 기록이 없으므로 `fsexactprog`의 과대 방출 표본이 논문 구성보다 적다(EXP-144: 5~12% 대 32~40%). 다만 한 시간 trace 후반에서 preemption이
  같았으므로(분당 19 대 19) admitted 차이는 작을 수 있다.
