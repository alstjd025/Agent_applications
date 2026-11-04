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

## 결과 A (2026-11-04 18:02~18:17, `261104_0103_exp146fixr1_fsexactprog_t75_rpm_2100`)

- 교체 +45 s. 새 스케줄러(`7dd6bf858b`) 목록의 남는 기록 **+51~+334 s 내내 0건** — EXP-145 19절의 경로(통지가 이전 스케줄러 파드로 감)는 시스템 수정으로 막혔다.
- 그러나 **+356 s부터 남는 기록이 생겼다**(43 → 525 → 782건, 측정 끝 강제 종료(+400~+430 s의 `server terminated` 1,554건)보다 앞). 그 기록들은 모두
  **+356 s 이전에 스케줄링되고 그 뒤에 끝났다.** +355 s는 이전 스케줄러 파드가 300 s 유예를 다 채우고 강제 종료된 시각이다(스택 snapshot: +55 s 스케줄러·게이트웨이 2개씩,
  +365 s 1개씩).
- 원인(코드로 확인): 게이트웨이의 `GatewayId`는 스케줄러와의 keepalive 연결의 로컬 주소(IP:임시 포트)다(`keepalive_client.go` `handShake`). 그 스케줄러 파드가 죽어
  연결이 다시 맺어지면 포트가 바뀌고, `NotifyComplete`는 "지금 주소 ≠ 스케줄링 때 주소"이면 통지를 보내지 않았다. **그래서 연결이 다시 맺어지기 전에 스케줄링되고
  그 뒤에 끝난 요청은 전부 통지가 빠진다.**
- 10월 run(EXP-141~143)의 100~200건은 이 경로로 더 잘 설명된다: 이전 스케줄러 파드가 측정 +65~+76 s(재시작 후 300 s 유예 끝)에 죽었고, 그때 처리 중이던 요청들의
  통지가 빠졌다. EXP-145 19절의 경로(16건)도 실재하지만 규모가 작다.
- 수정(바깥 `revision_scheduler` 다음 커밋): 통지 경로에서 주소 비교와 keepalive 준비 검사를 뺐다(통지는 요청 ID로 처리되고 HTTP로 간다). 게이트웨이 `05745270`
  (`staging/gateway-notifyfix2`). B가 도는 중이라 배포하지 않았다. B는 러너 수정으로 이전 파드가 측정 전에 제거되므로 이 경로의 영향을 받지 않는다.
- 운영 수정 확인: A의 측정 시작 시(-45 s snapshot) 스케줄러·게이트웨이 파드가 하나씩이었다.
