# EXP-141 — 처리 중인 요청을 관측으로 추적하면, 그리고 Llumnix\*가 처리 중인 요청의 가장 빡빡한 예산으로 판정하면 무엇이 바뀌나

세션 접두어: `exp141r`, `exp141hr`, `exp141base`, `exp141smoke`

작성 2026-10-20, **실행 전**. 코드는 바깥 저장소 브랜치 `revision_scheduler`(`bae80bb`), 설계 결정은
`project/revision.md`의 R30·R31과 "2026-10-19 — 처리 중인 요청을 관측으로 추적한다".

## 1. 무엇을 묻나

1. **(Q1) FluidServe의 장부 오차가 결과를 바꾸는가.** 논문 구성의 FluidServe는 처리 중인 요청을 추정한다(길이 분포 판정,
   그리고 장부 수가 엔진 running 수보다 많으면 먼저 보낸 요청부터 지움 — 35 req/s run에서 장부 이탈의 96%). 게이트웨이의
   완료 통지로 장부를 정확히 만들면(`--exact-residency`) 달성률·거절률·goodput이 달라지는가.
2. **(Q2) Llumnix\*의 판정 기준이 FluidServe와의 차이 중 얼마를 설명하는가.** Llumnix\*는 새 요청 자신의 TPOT 예산으로만
   판정한다. 처리 중인 요청 중 가장 빡빡한 명목 예산과 비교하게 바꾸면(`--slo-incumbent-tpot-gate`, llm-d·FluidServe와 같은
   형태) Llumnix\*가 FluidServe에 얼마나 가까워지는가.

## 2. 예상과 근거

- **Q1**: 차이가 작을 것으로 본다(모든 도착 기준 달성률 ±2점 안). 근거: 0.9 여유가 예측 오차(p99 4.5%)를 덮고 남고,
  장부 오차가 gate를 틀리게 만드는 것은 클래스가 섞인 인스턴스에서만이며, 35 req/s에서 chat 인스턴스는 거의 chat만 있다
  (10-10 재현 run의 인스턴스별 구성). 다만 정확 장부는 엔진 대기열의 요청까지 처리 중으로 세므로, 포화 구간에서 더
  보수적이 되어 거절이 늘 수 있다.
- **Q2**: Llumnix\*의 admitted 달성률이 오르고(느슨한 요청이 chat 인스턴스에 들어가 chat을 깨뜨리는 일이 줄어서) 거절률도
  오를 것이다. 모든 도착 기준 달성률은 FluidServe에 크게 못 미칠 것으로 본다 — 변형은 한 단계 예측(지금 상태 + 1)만
  보고, 같은 클래스끼리 모으는 배치·horizon 투영·pend가 없기 때문이다.

## 3. 무엇이면 틀렸다고 보는가 (판정 규칙, 실행 전에 고정)

채점 ladder95(`FS_SWE_TBT_MS=75 FS_SWE_TTFT_S=7`). 판정은 35 req/s 3회 평균으로 하고, 두 arm의 평균 차이가 두 arm의 반복
범위 중 큰 것과 **2점** 중 큰 값 이하이면 "차이 없음"이다(이 워크로드의 반복 간 편차 1.4~3.3점).

| 질문 | 틀렸다고 보는 조건 |
|---|---|
| Q1 | `fsexact`와 `fsv3capgnofrct75`의 모든 도착 기준 달성률 차이가 문턱을 넘는다(어느 방향이든). 넘으면 논문 수치가 장부 오차에 기대고 있다는 뜻이므로 R30을 논문 수준의 문제로 올린다 |
| Q2-a | `slot75inc`의 admitted 달성률이 `slot75`보다 문턱 이상 높지 않다 → 처리 중인 요청을 보호하는 판정만으로는 Llumnix\*의 SLO 위반을 줄이지 못한다 |
| Q2-b | `slot75inc`의 모든 도착 기준 달성률이 `fsv3capgnofrct75`와 "차이 없음"이 된다 → FluidServe의 이득이 판정 기준 하나로 설명된다(논문의 다른 메커니즘 주장이 약해진다) |

## 4. 정확한 설정

| 항목 | 값 |
|---|---|
| 함대 | Llama-3.1-70B × 4, TP=2, migration 끔(10-10 재구축 그대로) |
| 스케줄러 | `/home/nxclab/tools/staging/scheduler-residency2`, md5 **`33e7cc10`**(바깥 `revision_scheduler`, 장부 키 수정 커밋). 1단계의 확인 run과 `fsexact` smoke는 그 앞 빌드 `6b9b5eac`로 돌았다 — 두 빌드는 장부에 기록하는 인스턴스 키만 다르고, 그 둘은 그 키를 읽지 않는다(아래 5절 1단계 기록) |
| 게이트웨이 | `/home/nxclab/tools/staging/gateway-residency`, md5 `590242b0`, 모든 arm에서 `--notify-completion=true` |
| 드라이버 | 정적 `k8s/exp07/run_exp141_residency.sh`, 한 시간 `k8s/exp07/run_exp141_hour.sh`(EXP-108·109 드라이버의 복사본, 차이는 각 파일 머리말) |
| 워크로드 | 정적: FluidServe 계열 `mix_short_m1_t75.json`, Llumnix\* 계열 `mix_short_m1_t75fair.json`. 한 시간: `dyn60_shift_m2Am1B_b1045` 의 `_t75`/`_t75fair` |
| arm | `slot75`, `slot75inc`, `fsv3capgnofrct75`, `fsexact` |

## 5. 순서

1. **새 빌드 확인** — 플래그를 모두 끈 `fsv3capgnofrct75`, 35 req/s 8분 1회(`exp141base`). 기준: 10-10 재현 77.5 / 76.1과
   10-16 시연 77.2의 범위에서 2점 안. 벗어나면 새 빌드가 예전 동작을 내지 않는 것이므로 본 실험을 하지 않는다.
2. **smoke** — `fsexact` 4분, `slot75inc` 4분(`exp141smoke`). 볼 것: 스케줄러 지표 `scheduler_residency_requests`가 엔진의
   running+waiting 수와 같은 크기인가, `scheduler_fluidserve_retired_total{reason="completion"}`이 route 수만큼 오르는가,
   `scheduler_slo_incumbent_gate_total{binding="resident"}`가 0보다 큰가, `/complete` 실패 경고가 없는가.
3. **정적 본 실험** — 35 req/s 8분, 반복 3회가 바깥 루프, arm 넷을 안에서 번갈아(`exp141r1..3`). 약 3시간.
4. **한 시간 trace** — arm 넷 × 1회(`exp141hr1`). 1회라 판정에 쓰지 않고 경향만 본다. 약 5시간.
5. 끝나면 Sionna를 다시 켠다(`ms_dev/scripts/gpu_idle.sh start`). 다른 프로젝트의 실험이 10-21 01:50에 예약되어 있어
   그 전에 끝나야 한다.

## 6. 결과

### 1단계 (2026-10-20 11:29~12:17, 빌드 `6b9b5eac`)

- **새 빌드 확인** `261019_1938_exp141baser1_fsv3capgnofrct75_t75_rpm_2100`: 모든 도착 77.3 / admitted 99.9 / 거절 22.6% /
  goodput 13,790 tok/s, run 검사 7개 ok. 기준(10-10 재현 77.5·76.1, 10-16 시연 77.2에서 2점 안) 통과.
- **`fsexact` smoke** `261019_1953_exp141smoker1_fsexact_t75_rpm_2100`(4분): route 6,631건, 장부 이탈 이유 completion 6,425 /
  count 0 / survival 0 / age 0. 완료 통지 실패 경고 0.
- **`slot75inc` smoke** (4분): **기준이 한 번도 바뀌지 않았다** — `scheduler_slo_incumbent_gate_total{binding="own"}` 178,948,
  `resident` 0. 원인: 장부에 기록하는 키가 `LLMInstance.Id()`(`<ip>_instance<n>_<m>` 형식)였고 필터는 `GetInstanceId()`로
  조회한다. 수정해 `33e7cc10`으로 다시 빌드했고 `slot75inc` smoke를 다시 돈다(`exp141smokeb`). 이 run은 판정에 쓰지 않는다.

