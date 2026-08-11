# EXP-64 — what would per-request length knowledge buy?

Written before the run. 2026-08-07 01:50 KST.

## 1. The question

**We already predict; the question is the granularity.** Reading a class's
output-length distribution off past traffic and using it for the next request is
a prediction — it assumes the future looks like the past. What distinguishes us
from the systems we are compared against is not that they predict and we do not,
it is **how finely**: ours is conditioned on the class, Scorpio's on a 100-bin
classifier over the request, AdaGen's on a DistillBERT regression at 76.4%
accuracy, and SLOs-Serve assumes the decode length is known exactly.

*(This paragraph was rewritten on 2026-08-07. It previously said we have no
predictor, which is wrong and would have been read as a claim that we use no
historical information at all.)*

So a reviewer can ask two things and we can answer neither today:

1. **Would a finer-grained predictor help us?** If it would help a lot, the
   design is leaving something on the table and the paper should say so.
2. **Is our class-conditional estimate the reason we win?** If per-request truth
   adds nothing, then the answer is no, and the comparison against systems with
   finer predictors is not an information-advantage comparison.

**Building a predictor answers neither cleanly**, because a negative result would
be attributable to the predictor being bad. **Giving the policy the answer does.**
Whatever the gain from near-perfect per-request length knowledge turns out to be,
no predictor can beat it, so it is an upper bound on the whole question.

EXP-63 was built to measure the cost of the class-level estimate being *biased*
and did not manage it: scaling one class also re-weights the classes against each
other, so what moved was separation rather than accuracy (§65.7). Its replacement,
scaling all three classes together, is deferred. This experiment measures the
other half — the value of removing the estimate's *per-request variance* — and
carries the same hazard, which is why H3 below is a validity condition.

## 2. What "oracle" means here, exactly

The hint is **the output length the same request produced in an earlier run of
the same condition**, joined on `(task_id, call_index)`.

**It is not perfect and the paper must say by how much.** The same input
produces a different output length between runs even with a fixed seed, because
continuous batching changes the reduction order and flips near-ties: over 9,660
matched requests the inputs were 100% identical and the outputs were identical
**65.7%** of the time (chat 72.4 / deepresearch 44.2 / swe 40.9), with
`|difference|` p50 of 0 to 4 tokens and p90 of 28 to 122
(`fluidserve-implementation.md`, trap D).

So this is **a predictor about as accurate as a predictor could plausibly be**,
not an oracle in the strict sense. That is the honest description and it is also
the more useful one: it bounds what a real predictor could achieve.

**`(task_id, call_index)` is unique only in the eight-minute static conditions.**
On the one-hour trace 95,558 of 106,116 rows share that pair because tasks are
replayed hundreds of times, so this experiment is static-only.

## 3. How the hint reaches the policy — and why the workload does not change

The OpenAI completion API's **`user`** field is a free-form string that the
gateway already declares and re-marshals, and that vLLM accepts and ignores for
generation. The client packs `len:<tokens>` into it.

**This matters more than it sounds.** The obvious alternative, `max_tokens`,
would truncate generation, which changes the workload and makes the hint an
upper bound rather than a value. With `user` the two arms emit **byte-identical
requests** and differ only in whether the scheduler reads the field, so the
comparison has no workload confound at all.

| component | change |
|---|---|
| `pkg/types/scheduling_request.go` | `PredictedOutputTokens int` and a parser for the `len:<n>` encoding, rejecting anything else rather than guessing |
| `pkg/gateway/load-balancer/scheduler_client.go` | populate it from `cr.User` |
| `cmd/config/config.go` | `--fluidserve-oracle-length`, default **false** |
| `pkg/scheduler/policy/fluidserve.go` | when the flag is on and the request carries a positive hint, the request's expected remaining tokens and completion probability come from the hint instead of the class distribution — **for resident requests as well as the arriving one**, since the feasibility test is about what the incumbents still have to produce |
| workload | write `user="len:<n>"` from a lookup table built from a prior run |
| analysis | `ms_dev/scripts/build_length_oracle.py` builds the table from a result directory |

**Default off means the control arm is the same policy every experiment since
EXP-27 measured**, and a request without a parseable hint falls back to the class
distribution, so a missing table entry degrades to today's behaviour rather than
to zero.

## 4. Design

| | |
|---|---|
| rates | 45 and 60 req/s |
| arms | `fluidserve` (control, class distribution) and `fsoracle` (per-request hint) |
| repeats | 2 |
| conditions | 8, eight minutes each |
| mix | m1, stock engine ordering, migration off |
| oracle source | a prior `fluidserve` run at the same rate and mix, named in the experiment record |
| coverage check | the fraction of requests that carried a parseable hint, printed per condition; below 95% the condition is invalid |

Both arms send the `user` field. Only the flag differs. That way a mistake in
building the table cannot make the two arms differ in what they sent.

## 5. Hypotheses and judgement rules

### H1 — the gain is small

**Prediction**: `fsoracle` is within **3 points** of the control at both rates.

**Grounds**: the policy uses the length estimate through `E[L−j | L>j]`
multiplied by a pace, and the classes are chosen so that the within-class spread
is much smaller than the between-class spread (chat p50 389 / p90 766,
deepresearch p50 973 / p90 1246). If that is right, the class label already
carries most of the information and per-request truth adds little.

**Refutation, and the more consequential outcome**: the gain exceeds **5 points**
at either rate. Then per-request length prediction is a real lever we are not
pulling, the comparison against Scorpio and AdaGen becomes an
information-disadvantage comparison rather than the reverse, and the paper has to
say so. **Record it as a finding, not a failure** — it would point at a concrete
next version of the design.

### H2 — if there is a gain, it is concentrated in the class with the widest spread

**Prediction**: whatever gain appears, deepresearch shows more of it than chat,
because its p90/p50 ratio is the one the class mean serves worst in absolute
tokens.

### H3 — the mechanism is admission, not placement. **This is now a validity condition, not a hypothesis**

**EXP-63 showed why.** Scaling one class's length distribution moved offered
attainment by up to 5.6 points, and the movement was not accuracy at all: the
length profile is an input to "how loaded is this instance" as well as to "how
much more will this request produce", so inflating one class's lengths protects
the instances holding it and changes how strongly classes separate. Effective
instances per class tracked the scale factor at −0.897 and the score tracked
separation at −0.955 while demand did not move (§65.6, §65.7).

**Per-request truth does the same thing at request granularity.** A deepresearch
request whose real output is 1,400 tokens rather than the class mean of 985 makes
its instance look busier; one at 600 makes it look freer. So a gain here could be
accuracy or could be a separation shift, exactly as in EXP-63.

**Prediction, and the condition the run must satisfy to mean anything**: the
rejection rate moves and **effective instances per class move by less than 0.3**
between the two arms.

**If separation moves by 0.3 or more, this experiment did not measure accuracy
either**, and the answer needs the code split instead — a different length source
for `overIncumbents` than for the request's own feasibility test. Record that
outcome as a finding about the design rather than as a result about prediction.

**Prediction**: the rejection rate changes more than the per-engine class
composition. A better length estimate sharpens the feasibility test, which
decides whether a request can be served at all; it does not change which
instances a class prefers.

**Refutation**: the effective instances per class move by more than 0.3 while the
rejection rate does not move. That would mean the hint is acting somewhere other
than where it was wired in and the wiring must be re-read.

## 6. What this cannot answer

- **It does not deliver a predictor.** A deployed system would have to predict
  the length from the prompt, and the accuracy achievable there is 76.4% in
  AdaGen's measurement, well below the 65.7%-exact / few-token-error hint used
  here. **The result bounds what finer granularity is worth; it does not build
  it.**
- **It may not isolate accuracy**, for the reason in H3. The separation numbers
  have to be read before the score is interpreted.
- **It says nothing about the profile being biased**, which is EXP-63.
- **Static only**, for the join-key reason in §2.

## 7. 실행 전 갱신 — 무엇이 아직 맞고 무엇이 낡았나 (2026-08-11 확인)

**이 문서는 2026-08-07에 쓰였고 아직 한 번도 안 돌았다.** 그 사이에 워크로드가 바뀌었고(2026-08-08
부하 생성기 수정) 실험이 여덟 개 더 끝났다. **결과를 보고 규칙을 고치는 것이 아니라, 결과가 없는
상태에서 전제를 다시 확인한 것이다.**

### 7.1 아직 맞는 것

**전제(클래스 안의 퍼짐이 클래스 사이의 퍼짐보다 작다)가 수정 후 워크로드에서도 그대로다.**
§5가 근거로 든 값과 오늘 잰 값이 거의 같다(EXP-73의 `fspfx` @ 35 req/s, 2반복):

| class | §5가 인용한 값 | 2026-08-11 실측 | p90/p50 |
|---|---|---|---|
| chat | p50 389 / p90 766 | **p50 384 / p90 757** | 1.97 |
| deepresearch | p50 973 / p90 1246 | **p50 970 / p90 1219** | 1.26 |
| swe | — | p50 492 / p90 629 | 1.28 |

**`user` 필드로 힌트를 넣는 방법도 그대로 유효하다** — 두 arm이 바이트 단위로 같은 요청을 보내고
스케줄러가 그 필드를 읽는지만 다르므로 워크로드 교란이 없다.

**H3(분리가 움직이면 정확도를 잰 것이 아니다)은 여전히 유효 조건이다.** EXP-63이 그것 때문에
재려던 것을 못 쟀다.

### 7.2 ⚠ 오라클이 §2가 적은 것보다 정확하다 — 다시 쟀다

§2는 **65.7%가 완전히 같다**고 적었는데 그것은 **수정 이전 워크로드**의 값이다(함정 D).
수정 후 워크로드에서 같은 방식으로 다시 쟀다(EXP-73의 `fspfx` 두 반복을 `(task_id, call_index)`로
맞춰 붙임):

| | 짝지은 요청 | **완전히 같음** | \|차이\| p50 | p90 | 최대 |
|---|---|---|---|---|---|
| 25 req/s 전체 | 11,147 | **92.1%** | 0 | 0 | 623 |
| 　chat | 8,808 | 94.1% | 0 | 0 | 623 |
| 　deepresearch | 1,684 | 85.0% | 0 | 27 | 495 |
| 　swe | 655 | 84.7% | 0 | 16 | 509 |
| 35 req/s 전체 | 11,539 | **94.0%** | 0 | 0 | 905 |

**65.7% → 92~94%다.** 힌트가 예상보다 참값에 가깝고, **상한으로서 더 강하다** — "이보다 정확한
예측기는 없다"고 말할 때 그 상한이 더 높은 자리에 있다. §2의 65.7%는 **인용하면 안 된다.**

### 7.3 고쳐야 하는 설계 넷

**① 도착률 45와 60 → 25와 35로 바꾼다.** 60은 지금 격자(10·15·20·25·35·45·55·70)에 없고,
무릎이 28.1 req/s로 내려왔다. **그리고 EXP-78이 admission의 몫을 rate별로 쟀는데 45에서
0.8점이다**(모든 도착 분모). **H3이 "이 힌트는 admission을 통해 작용한다"고 말하므로, 45에서
재면 힌트가 작용할 자리가 거의 없는 곳에서 재게 된다.** 25(admission이 18.1점)와 35(4.4점)가
맞다.

**② 대조군 arm 이름이 바뀌었다.** §4가 `fluidserve`를 대조군으로 적었는데 **그 이름은 지금
prefix 회계를 끈 ablation**이다. 배포 기본 구성은 **`fspfx`**이고 대조군은 그것이어야 한다.

**③ H1의 "3점 이내"가 어느 분모인지 적어야 한다.** EXP-78 §7.7이 거절을 켜고 끄는 비교에서
분모가 결론을 바꾼다는 것을 보였다. 이 실험은 두 arm 다 거절을 켜므로 그 문제가 작지만,
**힌트가 거절률을 움직이는 것이 H3의 예측**이므로 **모든 도착 분모와 받아들인 것 분모를 둘 다
낸다.**

**④ 오라클 표의 출처**를 수정 후 run으로 정한다: **EXP-73의 `fspfx` @ 25·35 req/s**(2반복,
같은 arm·같은 mix·같은 바이너리 계열). §4가 "a prior `fluidserve` run"이라고 적은 것을 이것으로
바꾼다.

### 7.4 구현이 하나도 안 돼 있다

§3의 여섯 항목 중 **어느 것도 코드에 없다**(2026-08-11 확인: `--fluidserve-oracle-length`,
`PredictedOutputTokens`, `build_length_oracle.py`, 워크로드의 `user="len:<n>"` 전부 없음).

| 무엇 | 어디 | 크기 |
|---|---|---|
| `PredictedOutputTokens` + `len:<n>` 파서 | `pkg/types/scheduling_request.go` | 작음 |
| `cr.User`에서 채우기 | `pkg/gateway/load-balancer/scheduler_client.go` | 작음 |
| `--fluidserve-oracle-length` (기본 false) | `cmd/config/config.go` | 작음 |
| 힌트를 길이 분포 대신 쓰기 (**상주 요청도**) | `pkg/scheduler/policy/fluidserve.go` | **가장 큼** |
| `user="len:<n>"` 써 보내기 | 워크로드 | 중간 |
| 표 생성기 | `ms_dev/scripts/build_length_oracle.py` | 중간 |

**조건은 8개(2 rate × 2 arm × 2반복) 약 1.8시간**이고, 시간은 코드 쪽이 더 든다.

### 7.5 ⚠ 커버리지가 rate에 따라 갈린다 — 35 req/s는 사전 문턱을 못 넘는다 (2026-08-12 측정)

§4가 **"parseable 힌트를 가진 요청의 비율이 95% 미만이면 그 조건은 무효"**로 정해 두었다.
그 값을 실제로 재 보니 두 rate가 갈린다.

| rate | 표 출처 | 표 항목 | **커버리지** | 그 조건의 거절률 |
|---|---|---|---|---|
| 25 req/s | 앞선 run 1개 | 11,802 | **97.9%** | 1.9% |
| | 앞선 run 2개 합집합 | 11,979 | **99.3%** | |
| 35 req/s | 앞선 run 1개 | 13,491 | **80.0%** | 14.6% |
| | 앞선 run 2개 합집합 | 15,855 | **94.0%** | |

**이유는 표의 정의에 있다.** 표는 **출력을 낸 요청만** 담는다 — 거절되거나 실패한 요청은 길이가
없으므로 담을 수 없다. 그래서 **커버리지의 상한이 대략 (1 − 거절률 − 미완료율)**이고, 거절이
14.6%인 35 req/s에서는 한 run으로 80%가 최대다.

> ⚠ **그리고 빠지는 것이 하필 거절된 요청들이다.** 이 실험이 다루는 것이 판정 조건의 정확도인데,
> **판정에서 갈린 요청이 표에서 빠진다.** 즉 힌트가 있는 요청은 앞선 run에서 받아들여진 것들로
> 치우쳐 있고, 이 치우침은 합집합으로 줄일 수는 있어도 없앨 수는 없다.

**저부하 run으로 표를 만드는 길은 막혀 있다 (측정으로 배제).** 10 req/s는 거절이 없으니 거의
모든 요청의 길이를 담을 것 같지만:

| 저부하(10 req/s) 표를 쓰면 | 25 req/s | 35 req/s |
|---|---|---|
| 커버리지 | 40.3% | **28.8%** |
| 같은 요청의 길이가 완전히 같은 비율 | 62.5% | **62.1%** |
| \|차이\| p90 | 74 | 73 토큰 |

**커버리지가 낮은 이유**는 8분 동안 10 req/s가 만드는 요청이 4,800건뿐이라 35 req/s의 16,900건
대부분을 만난 적이 없어서다. **일치도가 낮은 이유가 더 중요하다** — 같은 부하 안에서는 92~94%가
완전히 같은데 부하가 다르면 62%다. **출력 길이가 부하에 의존한다.** 연속 배칭에서 배치 구성이
바뀌면 근소한 차이의 토큰 선택이 뒤집히는데, 부하가 다르면 배치 구성이 계통적으로 달라진다.

> **그러므로 오라클은 "같은 조건의 앞선 run"에서만 만들 수 있다.** 이것은 이 실험 설계의 제약이
> 아니라 **이 워크로드에서 "요청의 출력 길이"라는 양 자체의 성질**이고, 배포된 시스템이 예측기를
> 학습시킬 때도 같은 제약을 받는다 — **부하가 다른 구간에서 모은 표본은 62%만 맞는다.**

### 7.6 그래서 정하는 것

1. **표는 rate마다 따로, 그 rate의 앞선 run 두 개의 합집합으로 만든다.** 25에서 99.3%,
   35에서 94.0%.
2. **95% 문턱은 그대로 두고, 35 req/s가 그것을 못 넘는다는 사실을 결과와 함께 적는다.**
   문턱을 결과가 안 나온 상태에서 내리는 것은 규칙을 쉽게 만드는 변경이므로 하지 않는다.
   **35의 결과는 "커버리지 94.0%"를 옆에 달고 읽는다** — 힌트가 없는 6%는 대조군과 같은 동작을
   하므로, 그만큼 **처리군이 대조군 쪽으로 끌려간다.** 즉 **35에서 측정되는 이득은 과소평가**다.
3. **판정은 25 req/s에서 한다**(커버리지 99.3%, 그리고 EXP-78이 admission의 몫을 18.1점으로 잰
   자리). 35는 방향 확인용으로 같이 돌린다.

### 7.7 최종 설계 (2026-08-12, 실행 직전)

| | |
|---|---|
| 대조군 | `fspfx` (`oraclelen=false`) |
| 처리군 | `fsoracle` (`oraclelen=true`) — **다른 것은 이 플래그 하나뿐** |
| rate | **25 · 35 req/s** (둘 다 주 실험 격자 위. 무릎 28.1을 감싼다) |
| 반복 | 2 |
| 조건 | **8, 약 1.8시간** |
| 표 | `traces/length_oracle_m1_rpm_{1500,2100}.json`, EXP-73의 `fspfx` 두 반복의 합집합 |
| 커버리지 | 25에서 99.3%, 35에서 94.0% (§7.5) |

**양쪽 arm이 힌트를 똑같이 보낸다.** 워크로드는 표가 있으면 `user="len:<n>"`을 붙이고, 다른 것은
스케줄러가 그 필드를 읽는지뿐이다. 표를 잘못 만들어도 **두 arm이 보낸 것은 같다.**

**판정은 25 req/s에서 한다.** 커버리지가 99.3%이고, EXP-78이 그 도착률에서 admission의 몫을
18.1점으로 쟀으므로 힌트가 작용할 자리가 가장 크다. 35는 방향 확인용이고 커버리지 94.0%를 옆에
적는다.

**주 지표는 EXP-78·EXP-79와 같다**: 모든 도착이 분모, 거절과 미완료를 둘 다 위반. 받아들인 것
분모와 거절률을 같이 낸다. §5의 H1이 정한 3점·5점 문턱은 **이 지표로 읽는다**(§7.3의 ③).

**조건마다 확인하는 것**:
1. 기동 줄의 `oraclelen`이 의도한 값 — 검사기가 대조하고 실패하면 드라이버가 **멈춘다**(2026-08-11에 고쳤다).
2. `[oracle] worker coverage` 줄의 합이 25에서 95% 이상.
3. `client connection exhausted` 0건.

## 8. Result

(실행 후에 채운다)
