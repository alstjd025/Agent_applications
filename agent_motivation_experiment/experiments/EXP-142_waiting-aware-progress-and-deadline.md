# EXP-142 — 대기 중인 요청을 진행 추정에서 빼면, 그리고 첫 토큰 마감을 feasibility에 넣으면

세션 접두어: `exp142r`, `exp142smoke`

작성 2026-10-20, **실행 전**. EXP-141 r1에서 드러난 것의 후속이다.

## 0. 배경 (EXP-141 r1, 35 req/s 8분, 1회)

`fsexact`(게이트웨이 완료 통지로 장부를 정확히 유지)에서 deepresearch admitted가 99.9 → 62.6%로 떨어졌다. 실패 665건이
전부 TTFT > 10 s였고 926건 전부 인스턴스 8002에서 나왔다. 8002에서:

- 엔진 waiting 3.4 → 33.8, KV 96.9 → 99.6%, preemption 118 → 367
- 스케줄러의 KV outflow 예측 80K → 212K 토큰(2.7배), 투영 KV가 실제보다 21% 낮음, headroom −350K → +124K

원인은 둘로 본다.
1. **진행 토큰 수 j가 "보낸 뒤 엔진 step 수"라 엔진 대기열의 요청도 토큰을 만드는 것으로 센다**(`fluidserve_registry.go`
   `reconcile`). 옛 장부는 j가 큰 요청부터 지워 장부 수를 엔진 running 수에 맞췄으므로 이 오차가 outflow에 덜 들어갔다.
   정확 장부는 대기 중인 요청을 끝날 때까지 남기므로 outflow가 부풀었다.
2. **첫 토큰 마감이 feasibility에 없다**(R9, `--fluidserve-deadline-feasible=false`). 8002로 보낸 deepresearch의 예상
   첫 토큰 시간은 중앙값 14.3 s였고, TTFT를 넘긴 926건 중 905건은 스케줄러 자신의 예상으로도 10 s를 넘었다.

## 1. 무엇을 묻나

- **(Q1) A**: 엔진이 보고하는 대기 요청 ID(`InstanceStatus.waiting_requests`)로, 대기 중인 동안 j를 늘리지 않고 outflow에서
  빼면(`--exact-progress`) 정확 장부의 deepresearch 붕괴가 사라지는가.
- **(Q2) B**: 그 위에 첫 토큰 마감을 feasibility에 넣으면(`--fluidserve-deadline-feasible=true`) 더 나아지는가.

## 2. 예상

- Q1: `fsexactprog`의 deepresearch admitted가 논문 구성(`fsv3capgnofrct75`) 수준(99.9)으로 돌아온다. 근거는 0절 원인 1.
- Q2: 정적 35 req/s에서는 차이가 작을 것이다(A가 원인을 고쳤다면 8002 같은 대기열이 생기지 않아 마감 조건이 걸릴 일이
  드물다). EXP-100(8월, 옛 구성)에서 이 플래그는 deepresearch 위반량을 바꾸지 못했다.

## 3. 판정 규칙 (실행 전 고정)

ladder95, 35 req/s 3회 평균. "차이 없음" = 평균 차이가 두 arm의 반복 범위 중 큰 것과 2점 중 큰 값 이하.

| 질문 | 틀렸다고 보는 조건 |
|---|---|
| Q1 | `fsexactprog`의 deepresearch admitted 평균이 98 미만, 또는 `fsexactprog`와 `fsv3capgnofrct75`의 모든 도착 기준 달성률이 "차이 없음"이 아닌데 `fsexactprog`가 낮다 |
| Q2 | `fsexactprogdl`이 `fsexactprog`보다 모든 도착 기준 달성률이 문턱 이상 낮다(마감 조건이 거절만 늘린다) |
| 계측 | smoke에서 `scheduler_fluidserve_waiting_residents`가 0보다 커지지 않으면(대기 ID가 장부 ID와 맞지 않음) 본 실험을 하지 않는다 |

## 4. 설정

| 항목 | 값 |
|---|---|
| 스케줄러 | `/home/nxclab/tools/staging/scheduler-progress`, md5 `d2d6d184`(바깥 `revision_scheduler`, `--exact-progress` 추가). 새 플래그를 끄면 `33e7cc10`과 같은 동작 |
| 게이트웨이 | `590242b0`, `--notify-completion=true`(EXP-141과 같음) |
| 드라이버 | `k8s/exp07/run_exp142_progress.sh`(EXP-141 드라이버 복사본 + arm 둘) |
| arm | `fsv3capgnofrct75`(FluidServe 논문, 같은 세션 기준), `fsexactprog`(+ 정확 장부 + A), `fsexactprogdl`(+ B) |
| 조건 | 35 req/s 8분, 3회, 반복이 바깥 루프. 그 앞에 `fsexactprog` 4분 smoke |

## 5. 결과

(실행 후)
