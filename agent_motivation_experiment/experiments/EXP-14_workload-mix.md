# EXP-14 — 워크로드 mix rate sweep (chat + deep-research + SWE)

**날짜**: 2026-07-20 · **상태**: implementing · **브랜치**: `feat/exp07-kv-admission`

## 왜

지금까지는 워크로드를 **하나씩** 돌려 각각의 붕괴 특성을 확보했다:

| 워크로드 | 입력 mean | knee | 과부하 거동 | 정본 |
|---|---|---|---|---|
| chat (sharegpt) | 674 | ~55 req/s | tok/s 평탄 19.2k | EXP-12 |
| deep-research (searcharena) | 4,055 | ~13 req/s | tok/s 평탄 ~3.5k out | EXP-13 |
| SWE (codingagent) | 22,474 | ~5 call/s | **tok/s 붕괴** peak→1/3 | EXP-10 |

실제 서빙 클러스터는 이들이 **섞여서** 들어온다. mix에서 새로 물어볼 수 있는 것:

1. **간섭(interference)**: 무거운 SWE 요청이 같은 엔진에 들어왔을 때 가벼운 chat
   요청의 latency를 얼마나 망가뜨리나? (head-of-line blocking / KV 점유 경쟁)
2. **클래스별 공정성**: 과부하에서 어떤 클래스가 먼저·더 많이 SLO를 잃나?
   단일 워크로드 knee(55 / 13 / 5)로 예측한 것과 mix에서의 실제가 다른가?
3. **혼합 비율의 영향**: 무거운 클래스 비중이 커지면 전체 용량이 어떻게 무너지나
   — 요청 비율은 선형인데 토큰 질량은 비선형이므로.

## 설계

### 워크로드: `mixed_request_level_poisson` (신규)

세 request-level 워크로드를 **위임(delegate)** 방식으로 섞는 어댑터. 자체 요청
내용이 없고, 각 클래스의 단일 워크로드 의미론(프롬프트·결정성·절대 SLO·무-abort)을
그대로 유지 → mix 결과를 단일 워크로드 baseline과 직접 비교 가능.

- 클래스 배정: 가중치 합 크기의 **블록**을 결정적으로 셔플해 반복 → 목표 비율을
  블록 단위로 정확히 실현(i.i.d. 샘플링은 5분 run에서 비율이 흔들려 "비율 비교"라는
  실험 목적을 훼손). 자세한 근거는 `mixplan.py`.
- **metrics 스키마 무변경**: 각 delegate가 평소대로 기록(`agent=="request"`).
  클래스는 **task_id 접두사**로 복원 — `sg-`=chat, `sa-`=deep-research, 나머지=SWE.
  기존 분석 스크립트가 그대로 동작하고 per-class는 groupby로 얻는다.
- SWE transcript는 1.5GB라 24프로세스에 올릴 수 없어 **균등 샘플 1500건**
  (`transcript_swe_calls_mix1500.jsonl`, 170MB, 입력 mean 22,474 = 원본과 동일)을
  쓴다.

### 비율 (요청 수 기준)

**요청 비율 ≠ 토큰 비율**임에 주의 — 입력 크기가 클래스 간 ~33× 차이. run_config에
둘 다 기록한다.

| 조건 | mix (chat:dr:swe) | 요청 비율 | 예상 입력토큰 비율 | 의도 |
|---|---|---|---|---|
| **A** | 1:1:1 | 33/33/33% | 2.5/15/82% | 기준 — 균등 요청 |
| **B** | 6:3:1 | 60/30/10% | 8.6/26/65% | 경량 우세(현실적 프로덕션 믹스) |
| **C** | 1:1:3 | 20/20/60% | 1.2/7.2/92% | 중량 우세(에이전트 부하 스트레스) |

B는 chat이 다수인 현실적 구성, C는 SWE가 토큰 질량을 지배하는 스트레스 구성.
셋 다 토큰 질량은 SWE가 지배하지만 그 정도가 65%→82%→92%로 달라진다.

### 프로토콜 / grid

exp12-13 표준 동일: 조건별 cold restart, warmup 60s + 본 **5분** + grace 60s,
표준 분석 창 **[60,340]**, θ=0, no-timeout gateway + 64Gi.

- 부하기: **12 procs × 2048 threads** (transcript 170MB × 12 ≈ 2GB).
- **`--disable-timeouts` 필수**: SWE delegate만 `job_timeout_sec = baseline×tau`
  클라이언트 abort를 걸고 chat/deep-research는 abort가 전부 꺼져 있다. 그대로 두면
  mix에서 SWE만 죽어 클래스 간 비교가 오염된다(EXP-10도 같은 이유로 이 플래그를
  썼다). 이 플래그로 세 클래스 모두 무-kill로 통일한다.
- rate grid: 1:1:1 기준 요청당 평균 입력 ≈ (674+4055+22474)/3 ≈ 9.1k tok →
  EXP-13(4.1k에서 knee 13)과 EXP-10(22k에서 knee ~5)의 사이 → **knee 5–8 req/s 예상**.
  grid = **1, 2, 3, 4, 5, 6, 8, 10, 13, 16 req/s** (10조건, rpm 60~960).
- 소요: 조건당 ~10.5분 × 10 = ~1.75시간 / 비율. 3비율 = **~5.3시간**.

### 새로 붙는 계측: per-request → 엔진 귀속

EXP-13에서 불가능했던 "엔진별 SLO attainment"을 이번엔 측정한다. 구현 상세는
[DEV_request-engine-attribution.md](DEV_request-engine-attribution.md):
클라가 응답 `id`(uuid)를 `request_ids.jsonl` sidecar에 기록 + 수집기가 스케줄러의
`[Schedule] dispatch request <uuid> to ... instance <id>` 라인을
`server_metrics/scheduler_dispatch.log`로 스트리밍 → `build_request_engine_map.py`가
조인. mix에서는 이게 특히 의미 있다: **같은 엔진에 무거운 요청이 몰렸을 때 그
엔진의 가벼운 요청이 어떻게 되는지**를 직접 볼 수 있다.

## 가설

1. **간섭 가설**: mix에서 chat의 SLO knee는 단독(55 req/s)보다 훨씬 낮은
   전체-rate에서 무너진다. chat 요청 자체는 가볍지만 같은 배치/KV를 SWE와 공유하므로
   — 즉 **클래스별 knee는 자기 특성이 아니라 공유 자원 상태가 결정**한다.
2. **질량 지배 가설**: 전체 knee는 요청 비율이 아니라 **토큰 질량 비율**을 따라간다.
   A/B/C의 knee(req/s)는 SWE 토큰 지분(82/65/92%)의 역순 ≈ B > A > C 로 예상.
3. **붕괴 전염 가설**: SWE가 단독에서 보인 throughput 붕괴가 mix에서도 나타나되,
   SWE 지분이 클수록(C) 강하게 나타난다. A/B에서는 EXP-13처럼 평탄에 가까울 것.

## 결과

(실험 후 기록)
