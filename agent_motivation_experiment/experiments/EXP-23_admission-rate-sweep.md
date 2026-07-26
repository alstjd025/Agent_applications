# EXP-23 — 과부하에서의 admission: FluidServe 대 PolyServe rate sweep

**상태**: 설계 완료, 실행 대기 (EXP-22 v10 확인 후)
**브랜치**: llumnix `feat/fluidserve`, Agent_applications `feat/fluidserve`

## 왜 이 실험이 필요한가

EXP-22까지의 비교는 전부 **단일 부하 조건**이었다 — 고정 1800 rpm 2분, 또는 7분
동적 trace 하나. 이 조건에서 PolyServe의 admission은 사실상 발동하지 않는다.
동적 run에서 PolyServe가 거절한 요청은 **23,080건 중 20건**이었다.

즉 지금까지 측정한 것은 "거절이 없는 PolyServe 대 거절이 없는 FluidServe"이고,
두 정책의 **admission 계층 차이는 한 번도 측정된 적이 없다.** 두 정책이 다른
답을 내는 조건은 과부하이므로, 그 조건을 만들지 않으면 설계 차이가 결과에
나타날 수 없다.

## 두 admission의 차이 (측정하려는 것)

| | 거절 판단 기준 | 그 요청이 성공했을지를 보는가 |
|---|---|---|
| PolyServe | 인스턴스 KV 점유율이 임계값을 넘는가 | 안 본다 |
| FluidServe | **지금 가장 좋은 자리에 놓아도 이 요청이 자기 SLO를 못 지키는가** | 본다 |

FluidServe의 거절은 "이미 잃은 요청을 빨리 놓아주어, 아직 지킬 수 있는 요청의
자리를 비우는" 행동이다. 점유율 기반 거절은 아직 지킬 수 있었던 요청도 같이
버리고, 이미 못 지킬 요청도 점유율이 낮으면 받아들인다.

이 차이가 결과에 나타나는 방식에 대한 예측:

- **낮은 rate**: 둘 다 거의 다 지킨다. 차이 없음이 정상이고, 차이가 있으면
  둘 중 하나가 부하와 무관한 손해를 내고 있다는 뜻이다.
- **중간 rate**: 용량이 수요에 근접한다. FluidServe가 "지킬 수 있는 부분집합"을
  골라 지키는 이득이 나타나기 시작한다.
- **높은 rate**: PolyServe는 점유율이 임계값에 닿을 때까지 다 받아들이고, 받은
  요청은 대기열에서 예산을 다 쓴다. FluidServe는 못 지킬 요청을 미리 놓아주고
  나머지를 지킨다. 여기서 갈라져야 한다.

## 채점 — 분모를 offered로 고정한다

**거절·에러·미완을 전부 위반으로 센다.** 이유는 두 가지다.

1. **비교 가능성**: 거절을 분모에서 빼면 거절할 수 있는 정책이 자동으로 유리해진다.
   "지킬 수 없는 요청을 전부 거절하고 나머지만 세면 100%"가 되기 때문이다.
   FluidServe는 거절을 설계의 일부로 쓰므로 이 편향을 반드시 제거해야 한다.
2. **현실성**: 실제 서빙 시스템에서 거절된 요청은 사라지지 않는다. 클라이언트는
   재시도하거나 사용자가 떠난다. 어느 쪽이든 그 요청은 서비스되지 못한 것이고,
   분모에 남는 것이 정직하다.

집계는 **클래스 등가중**이다(chat, deepresearch, swe 각 1/3). fleet 전체 평균은
그때그때 많이 서비스된 클래스에 가중되므로 EXP-21이 기록한 구성 효과를 그대로
재현한다.

클래스별 규칙은 EXP-17/21과 동일:

| 클래스 | 규칙 |
|---|---|
| chat | 평균 TTFT ≤ 5s **그리고** 평균 TBT ≤ 50ms |
| deepresearch | 평균 TTFT ≤ 10s **그리고** 평균 TBT ≤ 100ms |
| swe | end-to-end ≤ 30s |

## 설계

```
./run_exp22_fluidserve.sh sweep "600,1200,1800,2400,3000,3600" 8
```

- **arm 2개**: polyserve, fluidserve. 스케줄러 정책만 다르다.
- **rate 6점**: 600 / 1200 / 1800 / 2400 / 3000 / 3600 rpm.
  600은 "둘 다 지킨다"를 확인하는 기준점, 1800은 EXP-21·EXP-22와 직접 비교되는
  점, 3600은 두 정책 모두 수요가 용량을 크게 넘는 점이다.
- **rate당 8분** + 조건마다 엔진 콜드 재시작 + 60초 저부하 예열(EXP-21과 동일).
  arm당 약 60분, 전체 약 2시간.
- 워크로드는 mix A(1:1:1), 엔진은 stock FIFO, migration off, KV threshold off —
  EXP-21/22와 같은 스택이므로 라우팅 정책만 비교된다.

## 필수 산출물

`analysis_scripts/request_level/exp22_report.sh`가 전부 만든다.

1. **rate 대 등가중 attainment 곡선** (arm 2개) — 주 결과. 어느 rate부터
   갈라지는지가 주장의 핵심이다.
2. **rate 대 클래스별 attainment** — 등가중이 어느 클래스에서 오는지.
3. **rate 대 token goodput**.
4. **rate 대 거절률** (두 arm). PolyServe의 거절이 언제부터 발동하는지, FluidServe의
   거절이 어느 비율인지.
5. **엔진 점유(engine_occupancy.py)** — 각 조건에서 fleet이 fleet으로 쓰였는지.
   한 엔진 최대 대기열과 idle-while-queued 비율. 이게 정상 범위를 벗어난 조건의
   attainment 값은 라우팅 결함의 결과이지 정책의 성능이 아니다.
6. **클래스별 엔진 분포** — 분리가 생겼는지.

## 미리 적어두는 실패 해석

- **모든 rate에서 FluidServe가 진다** → admission의 이점이 라우팅의 손해보다
  작다는 뜻이다. 클래스별 엔진 분포를 먼저 본다(분리가 없으면 라우팅 문제).
- **낮은 rate에서 지고 높은 rate에서 이긴다** → 설계대로다. 낮은 rate의 손해는
  과보수(불필요한 대기·거절)이므로 대기 마감과 거절 판정을 느슨하게 할 여지가 있다.
- **높은 rate에서도 안 갈라진다** → PolyServe의 admission이 그 rate에서도 발동하지
  않았을 수 있다. 거절률 곡선으로 먼저 확인한다. 발동하지 않았다면 rate를 더
  올리거나, 클러스터 용량 대비 수요가 부족한 것이다.
- **FluidServe의 거절률이 매우 높은데 attainment가 안 오른다** → 거절 판정이
  비관적이다. 거절된 요청이 실제로는 지킬 수 있었다는 뜻이므로, 예측 평균 step의
  편향(`predicted vs observed step` 계열, v9에서 p50 0.84 = 19% 낙관)을 본다.

## 관련

- 이 실험의 전제와 v9/v10 재설계: [EXP-22](EXP-22_fluidserve-routing.md),
  [fluidserve-implementation.md](../../../ms_dev/notes/fluidserve-implementation.md) §7
- 같은 스택의 rate sweep 선례: EXP-21 (PolyServe 대 stock Llumnix)
