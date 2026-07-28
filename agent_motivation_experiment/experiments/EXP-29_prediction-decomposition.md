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

## 7. 결과

(실행 후 기록)
