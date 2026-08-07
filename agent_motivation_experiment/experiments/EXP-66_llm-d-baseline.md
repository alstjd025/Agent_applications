# EXP-66 — llm-d predicted-latency를 기준선으로

**상태**: 실행 전. 이 파일은 **결과가 나오기 전에** 쓴다.
**배포·환경 계획은 `llumnix_reproduce/ms_dev/notes/llmd-baseline.md`가 정본**이고, 여기에는
가설과 판정 규칙만 둔다. 시스템 자체의 분석은 `related-works-review.md` §12.

---

## 0. 왜 이 실험을 하는가

llm-d v0.8의 predicted-latency scheduling은 **우리와 같은 계층에서 같은 형태의 판정 조건을
쓴다** — 새 요청의 예측 TPOT를 `min(그 요청의 예산, 그 인스턴스에서 이미 처리 중인 요청 중
가장 빡빡한 예산)`과 비교한다(`prediction.go:161-173`). 심사에서 "이미 있는 것 아닌가"로
나올 가장 유력한 후보이고, 지금 우리에게는 그것을 우리 워크로드 위에서 잰 값이 하나도 없다.

**그러므로 이 실험의 목적은 우리가 이기는 것을 보이는 것이 아니라, 차이가 어디에서
나오는지를 재는 것이다.** 판정 규칙에 "우리 주장을 좁혀야 하는" 결과를 먼저 적어 둔다.

---

## 1. 가설 — 실행 전에 적는다

| # | 가설 | 근거 | 무엇이 나오면 반증인가 |
|---|---|---|---|
| **H1** | `llmd-slo`가 `llmd-base`보다 높다 | 예산을 보는 것이 값을 만든다는 것이 우리 전체 주장의 전제다. 이것이 안 나오면 이 워크로드에서 예산 인식 자체가 값이 없다는 뜻이고, 그러면 우리 결과도 다시 봐야 한다 | 55·70 req/s에서 `llmd-slo` − `llmd-base` ≤ 0 이고 반복 범위가 겹친다 |
| **H2** | `llmd-pred`와 `llmd-slo`의 차이가 `llmd-base`와 `llmd-pred`의 차이보다 크다 | 예측을 갖는 것보다 그 예측을 예산에 대고 쓰는 것이 큰 일이라는 예상. 두 arm의 점수 방향이 반대이기 때문이다(§2.2) | 두 차이의 부호가 같고 크기가 뒤바뀐다 |
| **H3** | 세 llm-d arm의 클래스당 유효 인스턴스 수가 우리보다 4.0(균등)에 가깝다 | 그들에게는 클래스 개념이 없고 prefix 선호만 있다. **다만 우리 세 클래스는 프롬프트 구조가 크게 달라서 prefix 선호만으로도 갈릴 수 있다 — 그래서 이것이 가장 불확실한 가설이다** | llm-d arm의 클래스당 유효 인스턴스 수가 우리(chat 2.62 / dr 2.62 / swe 2.45, EXP-54 한 시간)와 0.3 안으로 붙는다 |
| **H4** | `llmd-slo`의 거절률이 0이 아니다 | 세 클래스를 전부 sheddable로 표시하고 InferencePool 경로로 돌리므로 거절 조건이 성립할 수 있다 | 경로 B에서도 거절이 0건 |

**H3가 반증되면 우리 주장을 넓혀야 한다** — "클래스를 봐야 분리가 생긴다"에서 "요청을 갈라
놓는 신호가 무엇이든 하나 있어야 한다"로. 그 경우 우리 기여는 분리의 존재가 아니라 **분리가
수요를 따라간다는 것**(motivation §8의 축 1)으로 좁혀진다.

---

## 2. arm

같은 엔진 넷(neutral-0의 vLLM 8000~8003), 같은 워크로드, 같은 믹스. 다른 것은 라우터뿐이다.

| arm | 라우터 | 구성 |
|---|---|---|
| `fluidserve` | Llumnix 게이트웨이 + 우리 스케줄러 | 배포 바이너리 `c2d970ca`, `FS_CLASS_HARM=false FS_FORCE_MARGIN=false FS_OWN_BUDGET_GATE=false` (EXP-27 이후 모든 조건과 같다) |
| `llmd-base` | llm-d EPP + Envoy | llm-d 기본 구성 — queue-scorer + kv-cache-utilization-scorer + prefix-cache-scorer + no-hit-lru-scorer. 예측 없음 |
| `llmd-pred` | 같음 | predicted-latency-producer + prefix-cache-affinity-filter + latency-scorer + weighted-random-picker. **SLO 헤더 안 보냄** |
| `llmd-slo` | 같음 | 위 + slo-headroom-tier-filter + latency-slo-admitter, `streamingMode: true`. **SLO 헤더 보냄**, 세 클래스 전부 sheddable |

### 2.1 SLO 헤더 매핑

| 클래스 | `x-llm-d-slo-ttft-ms` | `x-llm-d-slo-tpot-ms` | 출처 |
|---|---|---|---|
| chat | 5000 | 50 | 워크로드 설정 `slo` 블록 그대로 |
| deepresearch | 10000 | 100 | 그대로 |
| swe | 11800 | 25 | **채점 규칙은 전체 시간 30초인데 헤더로는 표현할 수 없다.** 워크로드 설정에 이미 있는 분해(11,800 + 25 × 728 = 30,000 ms)를 쓴다. 이 분해는 Niyama 이식 때 만든 것이고 새로 지어낸 값이 아니다 |

**논문에 적을 것**: swe의 전체 시간 예산을 TTFT + 토큰당으로 분해해 넣었고, 분해에 쓴 기대
출력 길이가 728 토큰이라는 것.

### 2.2 `llmd-pred`와 `llmd-slo`가 왜 방향이 반대인가

`latency-scorer`는 headroom = 예산 − 예측으로 점수를 만든다. 예산 헤더가 없으면 headroom이
전부 음수가 되고, 음수 구간에서는 `least`가 강제되는데 그것은 **위반 폭이 가장 작은 곳**,
즉 **예측 지연이 가장 짧은 곳**이다 → 부하를 퍼뜨린다. 예산이 있으면 양수 구간에서 `least`가
**예산 경계에 가장 가까운 곳**이 된다 → 부하를 예산 한계까지 모은다.
근거는 `scorer/latency/plugin.go:181-247`.

---

## 3. 조건

```
정적 sweep : 35 / 45 / 55 / 70 req/s
믹스       : m1 (mix_short_m1_balanced.json) — 지금까지의 정적 조건과 같다
반복       : 2회
조건 수    : 4 arm × 4 rate × 2 반복 = 32
조건당     : 예열 주행 3분 + 배수 대기(엔진이 빌 때까지, 상한 5분) + 본 측정 8분
소요       : 약 8시간
```

기존 arm(Llumnix, Llumnix SLO, PolyServe)은 EXP-53·EXP-57에 있으므로 다시 안 돌린다.
**세션이 다르므로 인용할 때 반복 간 편차를 밝힌다.**

### 3.1 조건마다 하는 일

```
① 엔진 재시작 (neutral-0 파드 삭제 → LWS가 다시 만든다) — 지금까지와 같다
② 예측기 파드 둘 재시작 → 빈 모델에서 시작
③ 예열 주행 3분 — 그 조건과 같은 도착률·믹스. 결과는 버린다
④ 배수 대기 — 엔진 넷이 전부 running=0 이고 waiting=0 일 때까지 (상한 5분)
⑤ 본 측정 8분
```

**③④를 `fluidserve`를 포함한 모든 arm에 똑같이 준다.** 예열 주행은 예측기 모델만이 아니라
**엔진의 prefix cache**도 데우고, prefix 일치율은 llm-d의 라우팅 입력이자 우리 분리 지표의
관측값이다. llm-d arm에만 주면 H3의 답이 오염된다.

⚠ **그래서 EXP-66의 `fluidserve` 값은 EXP-53·EXP-57의 것과 측정 절차가 다르다.** 같은 표에
놓을 때 적고, **EXP-66 안에서의 arm 간 비교를 주 결과로 삼는다.**

---

## 4. 판정 규칙 — 결과를 보기 전에 적는다

| 관측 | 결론 |
|---|---|
| `llmd-slo` < `fluidserve`, 차이가 반복 간 편차보다 크다 | 학습한 요청 단위 예측이 우리 방식보다 낫지 않다. **다음 질문은 §12.3의 차이 여섯 중 어느 것이 원인인가**이고, 그것은 이 실험이 답하지 않는다 |
| `llmd-slo` ≈ `fluidserve` (편차 안) | **우리 기여 주장을 좁혀야 한다.** 점수가 같다면 남는 것은 입력의 세기(그들은 요청 단위 학습, 우리는 클래스 조건부 분포)와 배포 비용이다 |
| `llmd-slo` > `fluidserve` | 요청 단위 예측이 값을 갖는다. EXP-64(길이 상한)가 그 상한을 재는 실험이 되고, 우리 설계에 예측기를 붙이는 것이 다음 방향 |
| `llmd-base` ≈ `llmd-pred` | 예측 자체는 값이 없고 예산을 보는 것이 값이다 (H2 지지) |
| `llmd-pred` ≈ `llmd-slo` | 반대로 예산·거절이 값이 없다 (H2 반증) |
| llm-d arm의 클래스당 유효 인스턴스 수 ≈ 4.0 | prefix 선호로는 클래스가 안 갈린다 → 우리 주장이 강해진다 (H3 지지) |
| llm-d arm의 클래스당 유효 인스턴스 수 ≈ 우리 값 | **prefix 선호만으로 같은 분리가 나온다** → 우리 주장을 넓혀야 한다 (H3 반증) |

⚠ **45 req/s는 반복 간 편차가 10점을 넘는 구간이므로 그 하나로 판정하지 않는다.**
55와 70 req/s가 판정의 기준이다.

⚠ **`llmd-pred`·`llmd-slo`의 점수를 인용하기 전에 예측 오차를 본다.** EPP가 예측과 실측을
히스토그램으로 낸다(`inference_objective_request_ttft_seconds` 대
`..._predicted_ttft_seconds`, 토큰당 시간도 같은 쌍). **3분 예열이 모자랐다면 본 측정 구간의
앞부분이 부하 균등화이고, 그것은 이 지표에 보인다.** 안 보고 점수를 쓰면 안 된다.

⚠ **arm 사이에 조건 시작 시각의 엔진별 prefix hit rate가 비슷한지 확인한다.** 안 비슷하면
③④의 통제가 실패한 것이므로 **H3에 답하지 않는다.**

---

## 5. 실행 전에 통과해야 하는 점검

| 단계 | 확인할 것 | 상태 |
|---|---|---|
| 0 | ghcr.io에서 이미지 넷을 받는다 | ✅ **2026-08-07 통과.** EPP 23 MB / Envoy 33 MB / 학습 617 MB / 예측 617 MB, 네임스페이스 `llmd`에서 확인 |
| 1 | 예측기 둘이 뜨고 예측 서버가 학습 서버에서 모델을 받는다 | ✅ **2026-08-07 통과.** 학습 1 + 예측 3 파드 Ready, 예측 서버가 `GET /model/{ttft,tpot}/info` 200. 학습 루프 1초 주기 확인 |
| 2 | 경로 A(file-discovery)로 EPP + Envoy가 뜨고 `curl` 한 번이 엔진에 닿는다 | ✅ **2026-08-07 통과.** 응답 200, 접근 로그에 처리 엔진이 찍힌다 |
| 3 | **요청 → 엔진 연결이 100%다** (Envoy 접근 로그의 `%UPSTREAM_HOST%`) | ✅ **2026-08-07 통과, 12/12 = 100%.** 엔진 넷에 3/4/3/3. **클라이언트 변경 불필요** — Envoy가 붙인 x-request-id를 vLLM이 completion id로 되돌려 주고 클라이언트가 그것을 이미 기록한다 |
| 4 | 헤더 전송을 껐을 때 `fluidserve` arm의 요청이 지금까지와 같다 | |
| 5 | 경로 B에서 **거절이 0이 아니다** | |

**단계 3이 가장 중요하다.** 여기서 100%가 안 나오면 H3에 답할 수 없고, 그러면 이 실험의
고유한 값(3·4번 질문)이 없어진다.

**단계 5가 0이면 멈춘다.** 거절이 안 되는 상태로 돌리면 `llmd-slo`가 아니라 `llmd-pred`에
tier 필터만 붙인 것을 재는 것이 된다.

---

## 6. 기록

| | |
|---|---|
| 배포·환경 계획 | `llumnix_reproduce/ms_dev/notes/llmd-baseline.md` |
| 시스템 분석 | `llumnix_reproduce/ms_dev/notes/related-works-review.md` §12 |
| 드라이버 | `k8s/exp07/run_exp66_llmd.sh` (아직 없음) |
| 연쇄 스크립트 | `/home/nxclab/tools/exp66_llmd.sh` (아직 없음) |
| 감시 | `/home/nxclab/tools/watch_experiment.sh` — 실험을 걸면 같은 턴에 건다 |
