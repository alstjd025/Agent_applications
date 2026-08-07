# EXP-67 — FluidServe가 prefix cache 재사용을 보면 어떻게 되는가

**상태**: 구현·단위 테스트 완료(`e594d8a`), **2026-08-08 02:00 KST에 실행을 걸어 둠**
(`/home/nxclab/tools/exp67_after_exp66.sh`, EXP-66 rep 2가 끝나면 자동 시작, 감시 동반).
**설계 정본**: [`ms_dev/notes/fluidserve-prefix.md`](../../../ms_dev/notes/fluidserve-prefix.md).
이 파일은 **실행 전에 적는 판정 규칙**과 조건 목록이다.

## 1. 왜

EXP-66에서 llm-d가 45 req/s 이상 전 구간에서 앞섰고(55 req/s offered 84.3 대 66.4),
원인을 축 여덟 개로 검사한 결과가 **클래스 분리가 아니라 prefix cache 재사용**이었다
(`llmd-baseline.md` §9.11). 70 req/s에서 엔진이 보고한 hit rate가 93.5% 대 75.1%이고,
실제로 계산한 prefill이 4,585 대 12,274 tok/s이며, 그 차이가 decode로 갔다
(20,661 대 16,562 tok/s).

우리 쪽 원인은 **task 단위 지역성**이다 — llm-d는 같은 task의 요청 99.9%를 같은 엔진으로
보내고 우리는 57.8~62.5%다. 클래스 선호는 클래스 수준의 지역성만 주고 그 아래에서 캐시가
깨진다.

## 2. 무엇을 바꿨나

도착 프롬프트의 prefill 비용이 `promptTokens × prefillFractionOf()`(fleet 스칼라 하나)에서
`(promptTokens − hitTokens(요청, 인스턴스)) × κ`로 바뀐다. **"몇 토큰을 계산해야 하는가"만
교체하고, "그 토큰이 엔진 시간으로 얼마인가"는 지금의 측정 경로(엔진의 대기 prefill,
측정된 duty cycle, 청크 모델)가 그대로 답한다.** KV 발자국은 할인하지 않는다.

κ는 지금의 `prefillFraction`인데 **분모가 바뀌어 뜻이 바뀐다**: `computed / Σ promptTokens`
(할인 그 자체)에서 `computed / Σ 예측 charge`(**그 예측의 오차**)로. 범위도 [0.02, 1.0]에서
[0.25, 4.0]으로 넓힌다 — 잔차는 1을 넘을 수 있어야 하고, 1을 넘는 경우가 바로 **색인이 엔진에
없는 hit을 주장한 경우**다.

## 3. ⚠ 먼저 정해야 하는 것 — 워크로드를 고칠 것인가

**아래 권고대로 진행했다 — 지금 워크로드로 EXP-67을 걸었고, 워크로드 수정은 적용하지 않았다.**
고치는 패치는 `/home/nxclab/tools/staging/fix_worker_prompt_duplication.py`에 준비돼 있고
`--check`로 적용 가능함을 확인했다. **워크로드를 고칠지, 고친다면 어느 arm을 다시 잴지는
사용자 결정이 필요하다**(§8.6에 비용). 2026-08-08에 부하 생성기의 결함을 찾았다:
`_worker_main`의 워커별 데이터셋 분할이 `isinstance(dataset, list)`로 갈라지는데 mixed
워크로드는 dict를 돌려주어 **분할이 통째로 건너뛰어졌고, 워커 12개가 같은 프롬프트 열을 그대로
보냈다.** 모든 프롬프트가 정확히 12번씩 나갔고 그만큼 prefix hit rate가 부풀려져 있다
(`fluidserve-prefix.md` §8).

| | 고치지 않고 EXP-67 | 먼저 고치고 EXP-67 |
|---|---|---|
| 비교 대상 | EXP-53/57/66과 나란히 놓인다 | **다섯 arm 전부 재측정 필요**(arm당 약 6시간) |
| 측정하는 조건 | prefix 재사용이 실제보다 큰 조건. **prefix affinity로 라우팅하는 정책에 유리** | 현실적인 조건 |
| 우리 변경의 이득 | 과대평가될 수 있다 | 제대로 잰다 |

**권고**: 두 단계로 나눈다. **EXP-67은 지금 워크로드로** 돌려 변경이 의도대로 동작하는지부터
확인하고(H1·H2가 그것을 잰다), 동작이 확인되면 **EXP-68에서 고친 워크로드로 FluidServe와
llm-d 둘만** 재서 방향을 본다. 이유: 워크로드와 정책을 같은 단계에서 바꾸면 둘을 구별할 수
없고, 이것은 이 저장소가 이미 겪은 실패다(프로파일과 워크로드를 따로 갱신하지 않아 생긴
CLAUDE.md 함정 A의 항목).

## 4. arm

| arm | 플래그 | 무엇을 가른다 | 이번에 도는가 |
|---|---|---|---|
| `fluidserve` | (기본, prefix off) | 대조군 | **돈다** |
| `fspfx` | `--fluidserve-prefix-aware=true` | 이 변경 전체 | **돈다** |
| `fspfx-nocal` | `+ --fluidserve-prefix-calibration=false` | κ가 무엇을 하는지 | 드라이버에 정의만 해 두고 **이번에는 안 돈다** |

**실제로 건 것: 2 arm × 3 rate(2700/3300/4200 rpm = 45/55/70 req/s) × 2 반복 = 12조건,
약 4.5시간.**

- **여덟 rate가 아니라 셋인 이유**: 45·55·70이 EXP-66에서 llm-d가 앞선 구간이고(+1.3, +17.9,
  +14.2), 동시에 feasibility가 구속력을 갖는 구간이다. 설계 §5대로 이 변경은 그 구간에서만
  작동할 수 있으므로, 15~35 req/s는 세 정책 전부 100점이라 읽을 것이 없다.
- **대조군을 같은 세션에 두는 이유**: EXP-53의 FluidServe를 재사용하면 절반 값이지만,
  **그 arm의 반복 간 편차가 60 req/s에서 11.7점**으로 이번에 읽으려는 효과보다 크다.
  같은 세션의 대조군은 세션 오프셋을 공유하므로 차분에서 상쇄된다.
- **⚠ 반복 2회가 모자랄 수 있다** — prefix 색인은 이력 의존 상태라 같은 설정의 두 run이 더
  갈라진다. **반복 간 편차를 먼저 읽고, 읽으려는 차이보다 크면 반복을 늘린다.**

## 5. 판정 규칙 — 실행 전에 적는다

| # | 예상 | 무엇이 나오면 틀린 것인가 | 어디서 읽나 |
|---|---|---|---|
| H1 | 엔진별 prefix hit rate가 75.1% → **85% 이상** | 안 오르면 신호가 결정에 도달하지 못한 것 | `server_metrics/engine_*.jsonl`의 `vllm:prefix_cache_hits_total / queries_total` |
| H2 | task당 유효 엔진 수가 2.08 → **1.6 이하** | 안 내려가면 순위가 여전히 클래스 항에 지배되는 것 | `analysis_scripts/request_level/separation_measures.py`, task 단위 |
| H3 | 45~70 req/s에서 offered 달성률이 **반복 간 편차보다 크게** 오른다 | 편차 안이면 이득 없음 | `exp53_compare.py` |
| H4 | κ가 **1.0 ± 0.25** 안에 머문다 | 1.5를 넘으면 축출을 못 따라가는 것 → 색인 용량이나 TTL을 손봐야 한다 | `scheduler_fluidserve_prefill_fraction` |
| H5 | `infeasible_total{reason}`가 gate/incumbents에서 **memory 쪽으로** 이동 | 안 움직이면 charge가 결정을 안 바꾼 것 | `scheduler_fluidserve_infeasible_total` |
| H6 | 클래스별 유효 인스턴스 수가 **0.3 이상 움직이지 않는다** | 움직이면 클래스 분리도 같이 바뀐 것이라 두 축을 못 가른다 | 같은 스크립트, 클래스 단위 |

**H1과 H2가 핵심이고, 둘 다 실패할 수 있다는 것을 미리 안다.** 이유는 설계 문서 §5에 적었다:
`sortCandidates`의 점수가 `w·share + (1−w)·room`이고 기본 w=1.0인데 **prefix는 그 어느 항에도
안 들어간다.** feasibility 판정과 TTFT 판정에만 들어가므로, 캐시가 있는 쪽과 없는 쪽이 둘 다
feasible이면 클래스 점유율이 이긴다. 즉 **지역성은 feasibility가 구속력을 가질 때만 생긴다.**

→ 둘 다 실패하면 다음 선택지는 `score`에 세 번째 항을 넣는 것인데, **그 순간 "예측을 정확하게
만든다"가 "새 목적함수를 넣는다"로 바뀌어 논지의 성격이 달라진다.** 이 실험에는 넣지 않고,
결과를 보고 따로 결정한다.

## 6. 사전 점검 — 조건을 걸기 전에

1. **배포된 바이너리가 새 플래그를 안다.** `--fluidserve-prefix-aware`가 `--help`에 있고
   기동 줄에 `prefix=true, prefixcalib=..., prefixblock=16, prefixcap=500000`이 찍히는지.
   ⚠ 드라이버가 플래그를 고정하면 그 바이너리에만 있어야 한다(CLAUDE.md 함정 B).
2. **대조군 arm이 정말 대조군인가.** `set_scheduler_profiling.py`가 이번 호출에서 설정하지
   않은 ablation을 제거하는지 확인하고, 기동 줄을 되읽어 `prefix=false`인지 본다.
3. **한 시간 trace의 서로 다른 프롬프트 수를 센다.** `--fluidserve-prefix-capacity` 기본값
   500,000이 충분한지는 8분 조건(0.30 M 블록)이 아니라 한 시간 조건으로 정해야 한다.
4. **감시를 같이 건다.** `/home/nxclab/tools/watch_experiment.sh` (CLAUDE.md 함정 B).
5. `kubectl get nodes` + `df -h`로 DiskPressure를 본다.

### ⚠ EXP-53과 다른 배경 하나

llm-d의 스택 다섯 파드(router/EPP, 학습 서버, 예측 서버 셋)가 **엔진과 같은 노드 nxc13에 떠
있는 채로** EXP-67이 돈다. EXP-53·EXP-57이 돌 때는 없던 것이다. 트래픽이 안 가므로 유휴에
가깝겠지만 **크기를 재지 않았다**(이 클러스터에 metrics-server가 없다).

**EXP-67의 판정에는 영향이 없다** — 대조군 `fluidserve`가 같은 세션에서 같은 배경으로 돌고,
읽는 것은 두 arm의 차이다. **영향이 있는 것은 EXP-67의 절대값을 EXP-53·EXP-66과 나란히 놓을
때**이고, 그때는 이 차이를 밝힌다. 내리지 않은 이유는 llm-d를 다시 재려면 다시 세워야 하고
그 비용이 이 배경 차이보다 크기 때문이다.

## 7. 결과

(실행 후 채운다)
