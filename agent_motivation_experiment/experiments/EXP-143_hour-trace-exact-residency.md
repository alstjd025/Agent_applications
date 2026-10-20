# EXP-143 — 한 시간 trace에서 처리 중 기준 Llumnix SLO와 고친 FluidServe

세션 접두어: `exp143hr`

작성 2026-10-20, **실행 전**. EXP-141에서 미룬 한 시간 trace(4절 4단계)를, EXP-142에서 고친 구성으로 바꿔 돈다.

## 1. 무엇을 묻나

정적 35 req/s에서 본 것이, 도착률과 클래스 비율이 시간에 따라 바뀌는 한 시간 trace에서도 같은 방향으로 나타나는가.

- (Q1) FluidServe + 정확 장부 + 엔진 대기 요청 제외(`fsexactprog`: 게이트웨이 완료 통지로 장부를 유지하고, 엔진이 대기 중이라고
  보고한 요청은 생성 토큰 수를 늘리지 않고 KV outflow 예측에서도 뺀다)가 논문 구성의 FluidServe(`fsv3capgnofrct75`)와 같은 수준인가.
- (Q2) Llumnix SLO + 처리 중 기준(`slot75inc`: 새 요청의 TPOT 판정을 자기 예산 대신 min(자기 예산, 그 인스턴스에서 처리 중인 요청의
  가장 빡빡한 명목 예산) × 0.9로 한다)이 정적 실험처럼 Llumnix SLO(`slot75`)보다 admitted가 낮고 거절이 적은가.

## 2. 예상

- Q1: 같은 수준. EXP-142 r1에서 deepresearch admitted가 99.95로 돌아왔다.
- Q2: 정적 실험(EXP-141 3회: admitted 38.3 → 25.8, 거절 43.0 → 8.6%)과 같은 방향.

## 3. 판정 규칙 (실행 전 고정)

**arm마다 1회이므로 통과·실패 판정을 하지 않고 경향만 기록한다.** 한 시간 trace에서 같은 arm의 반복 간 편차가 3.8점(EXP-109 계열)이므로,
ladder95 모든 도착 기준 차이가 4점 이하이면 "구별할 수 없음"으로 적는다. 결론이 필요해지면 반복을 더한다.
추가로 볼 것: 10분 구간별 클래스별 admitted, 그리고 FluidServe 두 arm의 deepresearch TTFT 초과 건수.

## 4. 설정

| 항목 | 값 |
|---|---|
| 스케줄러 | `3251836a`(바깥 `revision_scheduler` `7c5f8ad`, EXP-142 두 번째 시도와 같은 바이너리) |
| 게이트웨이 | `590242b0`, 모든 arm에서 `--notify-completion=true` |
| 드라이버 | `k8s/exp07/run_exp143_hour.sh`(EXP-141 한 시간 드라이버 복사본 + `fsexactprog` arm) |
| trace | `dyn60_shift_m2Am1B_b1045`(EXP-109와 같음). FluidServe 계열 `_t75`, Llumnix SLO 계열 `_t75fair` |
| arm 순서 (사용자 지정) | `slot75inc` → `fsexactprog` → `slot75` → `fsv3capgnofrct75`, 각 1회 |
| 시간 | arm 하나 약 75~80분. EXP-142가 끝난 직후(19:30 무렵) 시작, 01:45까지 끝나지 않을 arm은 건너뛴다(다른 프로젝트 실험 01:50 예약). 끝나면 Sionna |

## 5. 결과

(실행 후)
