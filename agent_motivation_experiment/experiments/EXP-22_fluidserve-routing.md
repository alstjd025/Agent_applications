# EXP-22 — FluidServe vs PolyServe on a dynamic trace

## Why

EXP-21은 라우팅 축에서 PolyServe가 엔진 스케줄러 5종을 전부 앞선다는 것을 보였고,
그 이득의 출처가 **tier 파티션**(무거운 클래스의 prefill이 가벼운 클래스의 배치에
섞이지 않게 하는 것)임을 admission 거부 사유 분석으로 확인했다. 다만 그 실험은
**정적 조건의 격자**였다 — 조건 하나가 rate 하나, 믹스 하나, 5분이다.

PolyServe의 tier→서버 배정은 10초 주기로 재계산되고 2회 연속 같은 값이어야 적용되며,
tier마다 최소 1대를 보장한다. 수요가 고정된 실험에서는 이 구조가 비용을 치르지 않지만,
수요가 움직이면 세 가지가 문제가 될 수 있다.

1. **반응 지연** — 배정이 바뀌려면 20~30초의 안정된 수요가 필요하다.
2. **최소 1대 보장** — 어떤 클래스의 수요가 0에 가까워도 서버 한 대가 그 클래스에
   묶여 있어 나머지 클래스가 쓸 수 없다.
3. **경계의 경직성** — 서버 단위로만 나눌 수 있어, 수요 비율이 4대로 나누어떨어지지
   않으면 남는 용량이 생긴다.

FluidServe는 파티션을 만들지 않고, 요청마다 **각 인스턴스가 앞으로 얼마나 더 받을 수
있는지**를 계산해 배치하며, 아무 데도 받을 수 없으면 게이트웨이에 붙잡아 둔다.
설계는 [fluidserve-design.md](../../../ms_dev/notes/fluidserve-design.md),
구현 결정 기록은 [fluidserve-implementation.md](../../../ms_dev/notes/fluidserve-implementation.md).

## Hypothesis

1. 수요가 움직이는 구간(믹스 전환 직후, rate 급상승 구간)에서 FluidServe가 PolyServe를
   앞선다. 재분할 대기 시간 동안 PolyServe는 이전 수요에 맞춰진 배정으로 운영된다.
2. 낮은 rate 구간에서는 차이가 작다. 두 정책 모두 여유가 있고, 파티션의 비용이
   드러나지 않는다.
3. swe의 절대 attainment는 두 arm 모두 고부하에서 낮다. FluidServe의 이득은 swe를
   구하는 것이 아니라, **예산을 이미 초과한 요청이 인스턴스 전체의 용량을 묶어두지
   않게 하는 것**에서 나온다.
4. token goodput에서 FluidServe가 앞선다. 예산 안에 들어올 수 없는 요청을 붙잡아
   두거나 뒤로 미루면 그 용량이 만족 가능한 요청으로 간다.

**반증 조건**: 위 1이 성립하지 않고 rate-binned 곡선이 두 arm에서 겹치면, 동적
수요에서도 파티션의 경직성은 비용이 아니며 FluidServe의 추가 복잡도는 정당화되지
않는다.

## Exact settings

- 엔진: Llama-3.1-70B-Instruct, TP2 × 4, max-model-len 40960, chunked prefill
  (`max_num_batched_tokens=8192`), async scheduling. **양쪽 arm 모두 stock FIFO**
  (`SCHED_EXTRA_ARGS=''`), **migration OFF**, KV admission θ=0(off).
  드라이버가 시작 전에 셋 다 검사하고 아니면 중단한다.
- arm은 스케줄러 라우팅 정책만 다르다.
  - `polyserve` — EXP-21과 동일 설정 (`--polyserve-tier-decode-tokens "25:728,50:386,100:275"`).
  - `fluidserve` — `--fluidserve-class-budgets "25:e2e:30000,50:decode,100:decode"`,
    horizon 100 step, z=1.65, alpha=1.0, 프로파일 `/profiling/fluidserve.json`.
- **게이트웨이 설정 차이(의도된 것, 해석 시 명시할 것)**: FluidServe는 "지금은 아무
  인스턴스도 받을 수 없다"를 엔드포인트 미반환으로 표현하고, 게이트웨이의 보유-재시도
  루프가 그 요청을 들고 있다가 다시 묻는다. 즉 **재시도 간격이 곧 FluidServe의 재결정
  주기**다. 기본값 1000ms는 인스턴스 상태가 바뀌는 시간 규모(step 20ms 내외)보다 훨씬
  거칠어서 100ms로 낮추고, 보유 상한은 워크로드에서 가장 큰 TTFT 예산(swe 11.8s)을
  덮도록 12s로 올렸다. PolyServe arm은 EXP-21 그대로 1000ms/5s로 되돌린다.
- 워크로드: `mixed_request_level_poisson`, `--mode trace-replay`,
  trace `traces/dynamic/canonical/dyn60_azure4d.csv` (1시간, Azure 모양 rate
  10~50 req/s, 믹스 A→C→B→A 15분 구간), config `workload_configs/mix_dyn60.json`,
  `--priority-mode deadline` (양쪽 arm 모두 packed priority 전송).
- 프로토콜: arm마다 콜드 재시작 1회, trace에 내장된 60s warmup, 부하 12 procs × 2048 threads.

### 채점 (EXP-21과 달라지는 부분)

규칙 자체는 EXP-17/21과 같다: chat TTFT≤5s & 평균 TBT≤50ms, deepresearch
TTFT≤10s & 평균 TBT≤100ms, swe E2E≤30s. 집계는 **클래스 등가중**.

**분모가 다르다.** 기존 `served_rows`는 거절된 요청을 분모에서 제외한다. 거절할 수
없는 정책만 비교할 때는 맞지만, 거절하거나 오래 붙잡아 둘 수 있는 정책에는 쓸 수
없다 — 놓칠 요청을 받지 않는 것만으로 수치가 올라가기 때문이다. 따라서

- **offered (본 지표)**: 분석 창에 도착한 **모든** 요청이 분모이고, 거절·에러·무응답은
  위반으로 센다.
- **served (참고)**: EXP-21 정의 그대로. 기존 결과와 나란히 놓기 위해 병기한다.

run 종료 시점에 아직 진행 중이던 요청은 결과가 정해지지 않았으므로 양쪽 arm에서
동일하게 제외하고, 그 수를 함께 보고한다.

### 사전 점검 — trace 순서 불변식

[DEV_dynamic-trace-mix](DEV_dynamic-trace-mix.md)가 경고한 실패 모드(파일 순서와 정렬
순서가 어긋나면 **모든 요청의 클래스가 조용히 잘못 붙는다**)를 실행 전에 확인했다.

두 trace 모두 **역행 쌍은 0개**이고 정확히 같은 값인 쌍만 있다
(smoke 30개 / 1시간 186개, 소수점 4자리 반올림 때문). 러너의
`arrival_trace.load_arrival_trace`는 `list.sort()`를 쓰고 파이썬의 정렬은
**안정 정렬**이므로 동점은 파일 순서를 유지한다. `mixplan.load_class_plan`도
역행만 거부하고 동점은 통과시킨다. 즉 클래스↔도착 짝은 보존된다.

## Result

(진행 중)

## 한계 / 해석 주의

1. **게이트웨이 재시도 설정이 arm마다 다르다**(위 참조). FluidServe에는 필수 배관이지만
   PolyServe에는 실패 경로일 뿐이다. 이 차이가 결과를 만들었는지 확인하려면
   "PolyServe + 100ms 재시도" arm을 추가해야 한다.
2. **PolyServe의 출력길이 파라미터가 실측과 다르다.** EXP-21이 쓴 `25:728,50:386,100:275`
   대비 실측은 swe 520 / chat 422 / deepresearch 282다(EXP-21의 10개 run, 60,042건).
   EXP-21 재현성을 위해 그대로 두었으나, FluidServe가 이겼을 때 이 파라미터 때문인지
   확인하려면 보정한 PolyServe arm이 필요하다.
3. **FluidServe의 용량 모델은 prefill 포화 구간에서 검증 표본이 적다.** φ≥0.6인
   window가 EXP-16 데이터에 40개뿐이다. 그 구간의 정확도는 온라인 교정과 안전계수가
   메운다.
4. **run이 arm당 1회다.** 1시간 run을 반복할 여유가 없으므로, 차이가 작으면 잡음과
   구분되지 않는다. rate-binned 곡선은 한 run 안에서 여러 window를 평균하므로 이
   점에서 단일 요약값보다 낫다.

## 산출물

- 드라이버 `k8s/exp07/run_exp22_fluidserve.sh`, 러너 `k8s/exp07/runner-dyn.template.yaml`
- 분석 `analysis_scripts/request_level/exp22_fluidserve.py`
- 결과 `results/aggregate_analysis/exp22/`
