# EXP-29 — 예측 반복시간이 실측보다 8 ms 높은 지점을 계측으로 특정한다

**상태**: 2026-07-29 진행 중
**브랜치**: `feat/fluidserve`
**바이너리**: `staging/scheduler-exp07-v24instr` (`md5 efce3b7e`) = v23 + 계측
**선행**: [fluidserve-implementation.md §24](../../../ms_dev/notes/fluidserve-implementation.md) (미해결 8 ms), §26 (두 가지 정정)

---

## 1. 무엇이 문제인가

80 req/s(4800 rpm)에서 FluidServe는 결정의 84%를 보유(PEND)하고, 배치된 것 중 62%가
약속을 깨고 밀어넣는 FORCE다. 그 사이 엔진은 KV 21%로 비어 있다.

| 값 | 측정 |
|---|---|
| 엔진 실측 반복시간 | 37.9 ms |
| 모델 예측(`f.meanStep`) | 45.9 ms |
| 게이트(`gateAfter`) | 45.0 ms |

예측이 게이트를 넘으므로 어떤 인스턴스도 feasible이 되지 않는다. **8 ms가 결정 전체를
뒤집고 있다.**

## 2. 왜 계측이 먼저인가 — 지금까지 세 번 틀렸다

이 간극에 대해 두 끝점(예측과 실측)만 보고 원인을 추정한 시도가 세 번 있었고 전부
틀렸다(§20.3, §22, §24.3). 예측과 실측 사이에는 원인이 될 수 있는 지점이 네 곳 있고,
바깥에서는 어느 것인지 구분할 수 없다.

```
meanStep = corr × [ dec + (sp/k)·(pre − c0) ]        ← 예측
measured = decodeOnly + duty_instant × measured      ← 실측 (duty의 정의)
```

- `corr` — 곱셈 보정계수 (`noteResidual`, 범위 0.5~3.0)
- `dec` — decode 법칙을 **현재 status**의 배치에서 평가한 값
- `decodeOnly` — decode 법칙을 **측정 구간에 실제로 돌던** 배치에서 평가한 값
- `(sp/k)·(pre − c0)` — prefill 항

### 대수적으로 보면 둘은 같아야 한다

`arrivingPrefill = duty × horizonMs / perChunk × chunk`이고
`horizonMs = k × pace`, `sp = arrivingPrefill / chunk`이므로

```
sp/k = duty × pace / perChunk
(sp/k)·(pre − c0) = duty × pace / perChunk × perChunk = duty × pace
```

`pace`는 실측 평균(`o.meanMs`)이므로 `pace = measured`. 따라서

```
meanStep / corr = dec + duty × measured
measured        = decodeOnly + duty × measured
```

**`dec == decodeOnly`이고 `corr == 1`이면 `meanStep == measured`가 항등적으로 성립한다.**
즉 8 ms는 다음 셋 중 하나(또는 조합)에서만 나올 수 있다.

| 후보 | 나타나는 방식 | 성격 |
|---|---|---|
| **A. `corr > 1`** | 45.9 / 37.9 = 1.21 부근 | 보정 루프가 1로 수렴하지 않음 |
| **B. `dec > decodeOnly`** | 두 status 사이 배치 증가 | 예측을 다른 상태에서 평가 |
| **C. `effectivePrefill = pendingPrefill > arrivingPrefill`** | 항등식이 깨짐 | 큐된 prefill이 지배 |

## 3. 계측 (이번에 추가)

| 지표 | 무엇 |
|---|---|
| `scheduler_fluidserve_decode_only_ms{instance}` | 실측 구간에 돌던 배치에서의 decode 법칙 |
| `scheduler_fluidserve_decode_law_ms{instance}` | 현재 status 배치에서의 decode 법칙 (`dec`) |
| `scheduler_fluidserve_obs_kv_tokens{instance}` | 그 법칙이 평가된 KV |
| `scheduler_fluidserve_obs_decode_batch{instance}` | 그 법칙이 평가된 요청 수 |
| `scheduler_fluidserve_pace_ms{instance}` | 지평 환산에 쓴 반복시간 |
| `scheduler_fluidserve_correction` | 곱셈 보정계수 (fleet 전역) |
| `scheduler_fluidserve_prefill_fraction` | 프롬프트 청구 비율 (fleet 전역) |

기존에 이미 있던 것: `observed_step_ms`, `predicted_step_ms`, `prefill_duty`,
`arriving_prefill_tokens`, `queued_prefill_tokens`.

**계측만 추가했고 결정 경로는 한 줄도 바꾸지 않았다.** 이 run은 v23의 동작을 그대로
재현해야 하며, 재현하지 않으면 계측 자체가 무언가를 바꾼 것이므로 그 run은 무효다.

## 4. 판정 규칙 (실행 전에 적는다)

`predicted − observed`의 8 ms를 세 성분으로 분해한다.

```
(1) corr 성분      = (corr − 1) × (meanStep / corr)
(2) 배치 성분      = corr × (dec − decodeOnly)
(3) prefill 성분   = corr × [(sp/k)(pre − c0) − duty × measured]
```

- **(1)이 6 ms 이상이면 원인은 보정 루프**다. `noteResidual`이 왜 1로 수렴하지 않는지를
  본다. 항등식상 고정점은 1이므로, 수렴하지 않는다면 duty의 EWMA 지연이나 `pace`가
  실측이 아닌 경로로 채워지는 경우를 의심한다.
- **(2)가 6 ms 이상이면 원인은 평가 상태**다. v23이 정의는 통일했지만 시각은 통일하지
  않았다는 뜻이 된다.
- **(3)이 6 ms 이상이면 원인은 `pendingPrefill`이 `arrivingPrefill`을 넘어서는 것**이다.
  `queued_prefill_tokens`와 `arriving_prefill_tokens`를 직접 비교해 확인한다.
- **어느 것도 6 ms를 넘지 않으면** 세 성분의 합이 8이 안 되는 것이므로 분해가 불완전한
  것이고, 그때는 원인을 더 추정하지 않고 분해식을 다시 세운다.

## 5. 과적합 경계 — 수정 단계에 미리 거는 조건

원인이 특정되어도 다음 세 조건을 통과하는 수정만 적용한다. §24.3에서 중간 지표를
결과의 대리로 삼아 세 번 틀렸으므로, 이번에는 **수정의 정당성을 숫자가 아니라 정의에서**
찾는다.

1. **정의로 정당화될 것.** "그 양이 무엇을 뜻하는가"로 설명되어야 한다. 상수 곱하기,
   오프셋 빼기, 계수를 결과가 좋아질 때까지 조정하기는 전부 기각한다.
2. **20/40/60 req/s에서 악화되지 않을 것.** 80만 좋아지는 수정은 이 rate에 맞춘 것이다.
3. **m2/m3에서도 성립할 것.** 정의 수정은 믹스가 달라도 성립하고, 맞춘 상수는 안 된다.

## 6. 실행

```
arm       fluidserve (v24instr)
mix       m1
rate      4800 rpm (80 req/s) 단일 — 문제가 나타나는 곳
duration  8분
반복       1회 (분해가 목적이므로 우열 판정이 아님)
session   exp29instr
```

---

## 7. 결과 (2026-07-29)

run `exp29b_fluidserve_m1_rpm_4800`, 계측 빌드 `3390cf3f`.

### 7.1 계측이 동작을 바꾸지 않았다 (사전 등록한 유효성 조건)

| | route | pend | shed | force | req-adm |
|---|---|---|---|---|---|
| v22 (pass 3) | 4.2 / 4.3 | 84.4 / 84.4 | 4.7 / 4.7 | 6.8 / 6.6 | 84.0 / 90.0 |
| v23+계측 | 4.3 | 84.3 | 4.5 | 6.8 | 85.1 |

조건 통과. 이 run은 v22/v23의 동작을 재현한다.

### 7.2 분해는 정확히 닫힌다

조립식 검증(`pred` 대 `corr × (dec + duty × pace)`)의 잔차가 평균 +0.20 ms,
중앙값 −0.02 ms다. **예측이 어떻게 만들어지는지에 대한 내 모델이 맞다**는 뜻이고,
아래 분해는 옳은 양을 분해한 것이다.

| 성분 | 평균 | 중앙값 |
|---|---|---|
| (1) 보정계수 (corr = 1.0220) | +0.84 | +0.85 |
| (2) 평가 배치 (`dec − decOnly`) | −0.06 | −0.00 |
| **(3) duty 항** | **−8.36** | **+5.77** |
| 잔차 | +0.20 | −0.02 |

**후보 A(보정 루프)와 B(평가 배치)는 죽었다.** v23이 배치 정의를 통일한 것은 옳았고
그 항은 이제 정확히 0이다. 전부 duty 항이다.

### 7.3 그리고 내가 쫓아온 8 ms는 통계량 불일치였다

같은 정책·같은 rate의 네 run이 전부 일치한다.

| run | obs 평균 | obs 중앙값 | pred 평균 | pred 중앙값 |
|---|---|---|---|---|
| pass3 r1 (v22) | 57.4 | 38.7 | 50.9 | 46.5 |
| pass3 r2 (v22) | 57.5 | 38.1 | 50.8 | 46.3 |
| v23 확인 | 56.5 | 38.8 | 50.4 | 46.2 |
| v23+계측 | 58.7 | 38.7 | 51.3 | 46.4 |

관측 반복시간은 **이봉분포**다: p5 30.8 / p25 34.9 / **p50 38.7** / p75 54.1 /
p90 98.2 / p95 168.3 / p99 395 / max 651. 65%가 게이트(45 ms) 아래이고 22%가 60 ms
위다. 8192 토큰 청크 하나가 약 436 ms이므로, prefill을 실은 반복이 수백 ms짜리
꼬리를 만든다.

**§24에 적은 "실측 37.9 / 예측 45.9 / 과대예측 8.0 ms"는 예측의 중앙값(46.4)을
관측의 중앙값(38.7)과 비교한 값이다.** 평균으로 비교하면 예측 51.3 대 관측 57.5로
**6.2 ms 과소**예측이다. 같은 데이터가 어느 통계량을 쓰느냐에 따라 반대 부호를 준다.
따라서 그 8 ms는 편향의 크기가 아니었고, 그것을 편향으로 보고 원인을 찾은 세 번의
시도는 **애초에 존재를 확정하지 못한 양을 쫓은 것**이었다.

### 7.4 duty 추정기의 실제 결함 — 서로 반대 방향의 둘

`duty_s = 0.163`(EWMA가 안정된 값), 표본별 순간 duty는 **평균 0.099 / 중앙값 −0.007**.

**① 표본마다 0에서 자르는 것.** 절반의 구간은 prefill이 전혀 없고 그 구간의 측정치는
decode 법칙 양쪽으로 흩어진다. `max(0, ·)`를 표본마다 적용하면 법칙보다 **느렸던**
구간은 남기고 **빨랐던** 구간은 버리므로, 결과는 평균이 아니라 **양의 부분의 평균**이
된다. 0.163 − 0.099 = 0.064, 즉 **추정치의 65%가 정류된 잡음**이다.

**② 구간을 같은 가중치로 평균하는 것.** prefill을 실은 구간은 수백 ms이고 decode만
있는 구간은 약 40 ms다. 두 구간의 비를 같은 가중치로 평균하면 "전형적인 한 구간 중
prefill 비중"이 되는데, `carriedPrefillTokens`는 이 값에 **밀리초 단위 지평**을 곱하므로
필요한 것은 **엔진 시간의 비중**이다. 시간 가중으로 다시 재면 같은 run이 **0.32**다.

두 결함은 상쇄되지 않고 **서로 반대 방향**이다 — 정류가 1.5배 올리고 균등가중이
절반으로 내린다. 그 결과가 관측 중앙값(38.7)과 관측 평균(57.5) **사이**에 앉았고,
그래서 한쪽 통계량으로 재면 8 ms 높고 다른 쪽으로 재면 6 ms 낮게 보였다. 이것이 이
간극이 세 번의 진단을 버틴 이유다.

### 7.5 수정 (v25) 과 사전 등록

두 시간을 각각 EWMA로 추적하고 **마지막에 나눈다.** 정류는 표본이 아니라 **비율에**
적용한다. 음의 prefill이 투영되지 않는다는 성질은 그대로 유지되고, 법칙이 편향되지
않았다는 증거인 구간들을 버리지 않는다.

```go
prefillMs := (measured - decodeOnly) * steps   // 음수 허용
totalMs   := measured * steps
o.prefillMsEwma = prev.prefillMsEwma + a*(prefillMs - prev.prefillMsEwma)
o.totalMsEwma   = prev.totalMsEwma   + a*(totalMs   - prev.totalMsEwma)
duty = clamp01(o.prefillMsEwma / o.totalMsEwma)
```

**측정 전에 적는 예상**: duty가 0.163 → 약 0.32로 오르고, 예측 반복시간이 관측
**평균**에 가까워지며, 게이트를 통과하는 배치가 **줄어든다.** 따라서

| 지표 | 예상 방향 |
|---|---|
| admitted 기준 attainment | **오른다** |
| 거절률 | 오른다 |
| token goodput | 내린다 |
| pend 비율 | 오른다 |

**이것은 최적화가 아니라 추정기 교정이다. attainment가 오르지 않으면 수정이 틀린 것이다.**
"80 req/s가 좋아졌다"는 결과만으로 채택하지 않는다 — §5의 세 조건(정의로 정당화,
20/40/60 악화 없음, m2/m3에서도 성립)을 전부 본다.

### 7.6 검증 run

`exp29v25`, m1, 20/40/60/80 req/s 전부. 80만 재지 않는 이유는 §5 조건 2 때문이다 —
진단한 rate에서만 좋아지는 수정은 그 rate에 맞춘 것이다.
