# EXP-27 — swe 입력을 줄이고, 믹스를 입력 토큰 기준으로 정의한다

**상태**: 설계 완료, 워크로드·설정·러너 준비 완료. EXP-25 종료 후 calib → sweep
**스크립트**: `k8s/exp07/run_exp27_mixsweep.sh`
**브랜치**: llumnix `feat/fluidserve` (v19+v20), Agent_applications `feat/fluidserve`

## 1. 왜 이 실험인가 — 지금까지 정적 파티션은 움직일 필요가 없었다

EXP-21~25는 전부 **요청 수 기준 1:1:1** 믹스로 측정했다. 그런데 클래스별 평균 입력이
chat 674 / dr 4,055 / swe 22,474 토큰이므로, 요청 수가 같으면 **입력 토큰의 82.6%가
swe 하나**에 몰린다. PolyServe의 수요 추정은 요청당 server-seconds에 도착률을 곱한
것이고 swe의 요청당 비용이 chat의 약 32배이므로, 어떤 비율을 줘도 수요가 swe에
지배된다.

`allocateServers`를 오프라인으로 재현해 확인했다(프로파일 테이블을 그대로 읽어
`serverSecondsPerRequest`와 largest-remainder 배분을 재구현; 스크립트는
scratchpad에 있고 결과만 옮긴다).

| swe 입력 | 믹스 (요청 수) | 입력 토큰 비율 chat/dr/swe | 배정 (swe/chat/dr) |
|---|---|---|---|
| 22,474 | 1:1:1 | 2 / 15 / 83 | **2 / 1 / 1** |
| 22,474 | 33:6:1 (토큰 1:1:1) | 32 / 35 / 33 | 1 / 2 / 1 |
| 22,474 | 8:1:1 (토큰 1:1:4) | 17 / 13 / 70 | **2 / 1 / 1** |
| **6,812** | 10:2:1 (토큰 1:1:1) | 31 / 37 / 31 | **2 / 1 / 1** |
| **6,812** | 40:2:1 (토큰 4:1:1) | 64 / 19 / 16 | 1 / 2 / 1 |
| **6,812** | 6:1:2 (토큰 1:1:4) | 19 / 19 / 63 | **2 / 1 / 1** |

즉 **믹스를 입력 토큰 비율로 정의하면 배정이 실제로 움직인다.** 요청 수로 정의하는
한 움직이지 않는다. 이것이 EXP-26(엔진을 3대로 줄여 격리를 불가능하게 만든다)을
하지 않고도 같은 질문에 답할 수 있는 방법이다.

## 2. 워크로드 변경 — swe 입력 22,474 → 6,812 토큰

`workloads/codingagent_request_level_poisson/build_short_transcript.py`가 기존
transcript를 변환해 만든다. 재실행 가능하고, 결과 파일은 git 대상이 아니다.

```
records                 1500
system prompt tokens    14,481 -> 4,047
conversation ratio      0.3694
mean input tokens       22,474 -> 6,812   (p10/p50/p90 = 4,227 / 6,569 / 9,578)
structural prefix share 81.5% -> 78.9%
```

**보존한 것과 그 이유**

| 보존 | 이유 |
|---|---|
| prefix 구조 | 모든 변환이 메시지 내용의 순수 함수라, 변환 전에 접두사를 공유하던 두 레코드는 변환 후에도 **동일한 토큰 접두사**를 공유한다. radix cache가 보는 공유 패턴이 규모만 줄어든 채 유지된다. 측정된 구조적 prefix 공유율 81.5% → 78.9% |
| 메시지 수와 역할 | 같은 agent job의 call 1, 2, 3 사이 접두사 관계가 유지된다 |
| 각 메시지의 끝 | 긴 메시지는 **가운데를 잘라낸다**. 모델이 응답할 stage 지시문이 마지막 user 메시지의 끝에 있으므로, 뒤를 자르면 모델이 하는 일 자체가 바뀌고 출력 길이도 바뀐다 |
| 길이 분포의 형태 | system 외 메시지에 단일 비율을 적용 |

**system prompt는 토큰 오프셋으로 자르지 않고 섹션을 골라 남겼다**(Role Definition /
General Principles / Tool Usage Instructions / Output Format Specifications + 마무리
문단). 실제 coding agent의 system prompt가 역할·도구·출력 형식으로 구성된다는 점에서
4,047 토큰은 현실적인 범위 안이고, 문장 중간에서 잘린 프롬프트를 남기지 않는다.

**보존하지 않은 것**: `baseline_ttft_s` / `baseline_tbt_mean_ms` / `baseline_e2e_s`는
원본 긴 프롬프트의 단독 실행 시간이며 짧은 버전에 대해 틀리다. Llumnix 믹스 실행은
`--disable-timeouts`로 `job_timeout_sec`을 0으로 만들고 절대 SLO 임계로 채점하므로
아무도 읽지 않는다. 스키마를 유지하기 위해 필드는 남기고 `baseline_valid: false`를
찍는다.

**KV hit rate가 비현실적으로 높아지지 않는가**: 반대다. 원본이 81.5%로 더 높았고,
그 대부분(14,481/22,474 = 64.4%)이 **모든 요청이 공유하는 하나의 system prompt**에서
왔다. 짧은 버전은 78.9%로 약간 낮다.

## 3. 세 가지 믹스

`workload_configs/mix_short_{m1_balanced,m2_chatheavy,m3_sweheavy}.json`

| 키 | 요청 수 chat:dr:swe | 실현 입력 토큰 비율 | PolyServe 배정 |
|---|---|---|---|
| m1 balanced | 10 : 2 : 1 | 31.1 / 37.4 / 31.4 | 2 swe / 1 chat / 1 dr |
| m2 chat-heavy | 40 : 2 : 1 | 64.4 / 19.4 / 16.3 | **1 swe / 2 chat / 1 dr** |
| m3 swe-heavy | 6 : 1 : 2 | 18.6 / 18.7 / 62.7 | 2 swe / 1 chat / 1 dr |

SLO 블록은 EXP-21/25와 동일하다(chat 5000ms/50ms, dr 10000ms/100ms, swe e2e 30s).
바꾸지 않는 이유는 채점 기준을 이전 실험과 이어 붙이기 위해서다. swe 입력이 3.3배
짧아졌으므로 swe는 **쉬워진다** — 그것이 목적이다. EXP-24에서 swe 달성률이
1800 rpm에 5.6%, 3000 rpm에 1.5%였는데, 그 상태에서는 등가중 지표의 1/3이 상수 0이라
정책 간 차이를 담지 못한다.

## 4. 채점 규약을 바꾼다

지금까지는 **offered 분모**만 봤다(거절·에러·미완을 전부 위반으로). 그 규약에서는
거절하는 정책이 거절하지 않는 정책을 이길 수 있는 구간이 원리적으로 없다.

이제 **두 분모를 나란히 보고한다.**

| | 정의 | 무엇을 왜곡하나 |
|---|---|---|
| **admitted** (주 지표) | 분모 = 시스템이 받아들인 요청 | 단독으로 읽으면 전부 거절하는 정책이 최고점을 받는다 |
| offered | 분모 = 도착한 모든 요청 | 어차피 못 지킬 요청을 거절해도 손실로 계산한다 |

**단독으로 읽지 않기 위한 두 개의 동반 지표**: 거절률(클래스별)과 token goodput.
거절된 요청은 토큰을 만들지 않으므로 과도한 거절은 goodput에서 반드시 드러난다.
목표는 **admitted 기준 모든 rate에서 90% 이상, 동시에 goodput이 PolyServe 이상**이다.

> 구현 결함 하나를 같이 고쳤다. `attain()`이 served 열에서도 거절된 요청을 분모에
> 남겨두고 있었다(거절된 요청은 first token이 없으므로 `miss`가 참). **2026-07-28
> 이전에 기록된 "served" 수치는 전부 사실상 offered 수치다.**

## 5. 설계

```
calib   polyserve만, m1/m2/m3, rates 600,1200,2400,3600, 조건당 4분   (~1시간)
sweep   rep(바깥) x mix(3) x arm(2), rates는 calib에서 정한 3점, 조건당 8분
```

- **반복이 바깥 루프**다. EXP-24에서 세션 내 산포 0.2점, 세션 간 이동 5점이 측정됐다.
  arm을 바깥에 두면 그 5점이 arm 효과로 읽힌다.
- 조건마다 엔진 콜드 재시작(`--restart-per-condition`).
- calib를 먼저 도는 이유: swe가 짧아져 fleet 용량이 바뀌었고, PolyServe의 수요 모델로
  추정한 값(m1에서 1,043 rpm)은 chat의 용량을 실측보다 낮게 잡는다는 것이 이미
  알려져 있다(3000 rpm에서 chat 100% 달성). **추정으로 rate 리스트를 정하면 전
  구간이 포화 후 구간이 되어 EXP-23~25를 이름만 바꿔 반복하게 된다.**

## 6. 실행 전에 적어두는 판정 규칙

| 관측 | 결론 |
|---|---|
| m2에서 PolyServe의 배정이 실제로 (1,2,1)로 움직임 | 이 워크로드가 원안이 말한 조건을 만든다는 확증. `scheduler_polyserve_tier_servers`로 확인 |
| 배정이 움직이는 동안 PolyServe의 chat/dr이 떨어짐 | 정적 파티션의 전환 비용이 실측됐다는 뜻. 마이그레이션이 꺼져 있으므로 재배정된 서버는 in-flight 요청이 끝날 때까지 두 클래스를 동시에 들고 있다 |
| FluidServe가 admitted ≥ 90%이면서 goodput ≥ PolyServe | 목표 달성 |
| admitted ≥ 90%인데 goodput이 낮음 | 여전히 과도하게 거절하는 것이다. 거절률과 클래스별 거절률을 본다 |
| 세 믹스 전부에서 배정이 안 움직임 | 오프라인 재현이 틀렸다는 뜻. `serverSecondsPerRequest`의 실측 입력(클래스별 평균 입력 토큰)이 내 가정과 다른지부터 확인 |

## 7. 함께 확인할 것

- `scheduler_polyserve_tier_servers` / `scheduler_polyserve_tier_demand` — 배정이
  움직이는 시점과 횟수
- `scheduler_fluidserve_prefill_duty{instance}` — v19가 넣은 계열. 인스턴스마다
  갈라져야 한다. 네 인스턴스가 같은 값이면 라우팅이 클래스를 섞고 있다는 뜻이다
- `engine_occupancy.py` — 엔진이 놀고 있는가
- `slo_rule_breakdown.py` — TTFT 실패인가 TBT 실패인가

## 8. 관련

- 선행: [EXP-25](EXP-25_routing-vs-admission.md) — 라우팅과 admission 분리
- 대안이었던 것: [EXP-26](EXP-26_partition-must-move.md) — 엔진을 3대로 줄이는 안.
  사용자 지시로 보류했고, 이 실험이 같은 질문에 워크로드 쪽에서 답한다
- 정책 변경 근거: `ms_dev/notes/fluidserve-implementation.md` §15 (v19), §16 (v20)
