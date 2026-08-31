# EXP-108 — 정적 rate sweep을 swe의 per-token 예산 형태로 다시 잰다

**상태**: 2026-08-31 09:02 KST 시작, 도는 중 (64조건, 약 19시간 예상)
**연쇄**: `/home/nxclab/tools/exp108_chain.sh`, 로그 `/home/nxclab/tools/exp108.log`
**드라이버**: `k8s/exp07/run_exp108_t75.sh`(llumnix 쪽 셋), `k8s/exp07/run_exp108_llmd.sh`(llm-d)
**바이너리**: `bin/scheduler-exp07` = `6dc9f035` (파드 안 `md5sum /proc/1/exe`가 유일한 authority).
이 하나에 EXP-107T의 FluidServe 플래그와 EXP-106의 PolyServe 스위치 여섯이 **둘 다** 들어 있다.
`/home/nxclab/tools/staging/scheduler-polyserve-paper`(`47f3e2b5`)는 EXP-107보다 앞선 판이라
FluidServe 플래그가 없고, 배포하면 FluidServe arm이 pflag 미지 플래그로 기동 중 종료한다.

---

## 1. 무엇을 묻는 실험인가

**swe의 약속을 전체 시간에서 토큰당 시간으로 바꾸면, 정적 도착률 곡선에서 네 정책의 순서와
간격이 어떻게 달라지는가.**

FluidServe v0.4가 그 전환을 했다(`ms_dev/notes/fluidserve-v0.4.md` §1.3). 이유는 성능이 아니라
**약속과 집행이 다른 양이라서**다: admission은 검사 시점의 순간 속도를 집행하는데 전체 시간
예산은 요청 수명에 걸친 적분이고, EXP-107 §10.2가 그 틈을 측정했다 — 승인 하나하나는 정직한데
(검사 시점 7초 창 예측이 예산 69.2 ms 이하) 함께 승인된 것들이 이후 수십 초에 걸쳐 만드는
합동 작동점이 78~80 ms가 된다.

**그런데 그 전환은 지금까지 우리 arm에서만, 그것도 한 시간 trace에서 반복 하나로만 이뤄졌다**
(v0.4 §5.1이 그 점을 미해결로 적고, §5.4가 정적 sweep 미확인을 적는다). 기준선 셋은 여전히
옛 형태로 판정받는다. **형태가 다른 두 시스템을 한 표에 놓으면 그 표는 정책 차이가 아니라
예산 차이를 보여준다.**

## 2. 무엇을 바꿨나 — 그리고 왜 워크로드 파일이 둘인가

swe = **TTFT 7초 + 토큰당 75 ms**. 75는 분포에서 유도한 통계가 아니라 chat의 50과 같은 지위의
요구사항 선언이다(v0.4 §1.3).

정책이 그 값을 어디서 받는지가 갈린다.

| 정책 | 예산을 어디서 읽나 | 이번에 주는 파일 |
|---|---|---|
| FluidServe | `--fluidserve-class-budgets 25:decode:75`. **25는 tier 키**이고 프로파일 조회와 모든 로그가 그 값으로 클래스를 부른다. 실예산은 셋째 필드 | `mix_short_m1_t75.json` (swe `ttft_ms 7000`, `tbt_ms` **25**) |
| PolyServe / llm-d / Llumnix SLO | 워크로드 설정의 `slo.<class>.tbt_ms`를 **토큰당 목표로 곧이곧대로** 읽는다. 셋 다 전체 시간 예산을 표현할 수단이 없다 | `mix_short_m1_t75fair.json` (swe `ttft_ms 7000`, `tbt_ms` **75**) |

**파일 하나로 통일할 수 없는 이유**: FluidServe 쪽 파일에 75를 쓰면 요청이 tier 75를 달고
나가는데 예산 맵의 키가 25라 매칭이 안 되고, 반대로 나머지 셋에 25를 주면 swe를 진짜 예산의
1/3로 판정한다(프로파일 표를 풀면 요청당 KV 7,306 토큰에서 25 ms는 배치 23까지, 75 ms는 260까지
허용한다 — `polyserve-fidelity.md` §9.5). 지금 `m1`(FluidServe)과 `m1f`(SLO·llm-d)가 갈려 있는
것과 같은 구조다.

두 파일 모두 `slo` 블록 밖은 `mix_short_m1_balanced.json`과 바이트 동일함을 확인했다 — 도착
과정·클래스 비율·프롬프트 풀·`num_conversations`가 같으므로 **부하는 같고 정책에게 알려주는
swe 예산만 다르다.**

**채점도 같이 움직인다**: `FS_SWE_TBT_MS=75`. `exp22_fluidserve.py`가 `SLO_RULES["swe"]`를
`{ttft: 7.0, tbt: 75.0}`로 바꾸고 `FS_SWE_E2E_S`와 상호배타이며 import마다 경고를 찍는다.
`all_arrivals_attainment.py`를 비롯한 나머지 스크립트가 거기서 `attain`을 가져다 쓰므로 형태
전환이 한 군데에서 일어난다.

## 3. arm

| arm | 정책 | 워크로드 | 무엇인가 |
|---|---|---|---|
| `fsv3capgnofrct75` | fluidserve | t75 | v0.4 채택 구성: guardrail cap on(`FS_INSTANCE_CAP=true`, `FS_CAP_WINDOW_MULT=3.0`) + force off(`FS_FORCE=false`) + per-token swe(`FS_SWE_TBT_MS=75`). **이름은 새로 짓지 않았다** — 한 시간 run `260827_2320_exp107tr1_fsv3capgnofrct75_shift`가 이미 그 이름이고, 정적 곡선과 한 시간이 같은 arm으로 읽혀야 한다 |
| `llmdslot75` | llm-d (별도 스택) | t75fair | 상류 `predicted-latency-slo` 구성, 경로 B(InferencePool + `sheddable` InferenceObjective)라 거절이 가능하다 |
| `polyservept75` | polyserve | t75fair | EXP-106이 되살린 논문 메커니즘 여섯 전부 on |
| `slot75` | slo | t75fair | Llumnix의 SLO 인식 변형, 변경 없음 |

**⚠ 이름의 `t75`는 장식이 아니다.** `run_exp106_polyserve.sh`의 `polyservep`는 **같은 여섯
스위치를 m1f(swe 52 ms)에서** 돌린다. 한 이름이 둘을 덮으면 같은 표에 못 들어가는 두 벌이
서로의 반복처럼 보인다. `slot75`도 같은 이유다 — 기존 `slo` 열은 52 ms로 측정된 것이다.

**⚠ 인용 주의**: 이 sweep의 어느 열도 `paper_experiment/static_sweep_2026-08`(=
`static_sweep_clean_2026-08`)의 같은 이름 열과 한 표에 들어갈 수 없다. 그쪽은 swe가 e2e 30초
(FluidServe·PolyServe·vLLM router는 m1, Llumnix SLO·llm-d는 m1f)로 판정된 것이다.
`retracted.tsv`에 넣는 것은 대체값이 다 나온 뒤에 한다.

## 4. 조건

정본 정적 세트와 **도착률·길이·반복이 같다**(`paper_experiment/static_sweep_2026-08/manifest.tsv`
에서 직접 확인): 10·15·20·25·35·45·55·70 req/s(600~4200 rpm), 조건당 **8분**, 조건마다 엔진과
제어평면 cold restart, **2반복**. 4 arm × 8 도착률 × 2반복 = **64조건**.

**반복이 바깥 루프이고 arm이 안쪽이다.** 순서는 `fsv3capgnofrct75 → llmdslot75 →
polyservept75 → slot75`. 이유 둘: 19시간에 걸친 기계 상태의 이동이 특정 arm에 몰리지 않고,
반복 하나가 끝날 때마다 네 arm의 도착률 곡선이 한 벌씩 완성되어 중간에 읽을 수 있다.

## 5. 실행 전에 적는 판정 규칙

### 5.1 유효성 검사 — 하나라도 실패하면 그 조건은 판정에 쓰지 않는다

1. **FluidServe arm의 기동 줄**에 `budgets "25:decode:75,50:decode,100:decode"`,
   `instancecap=true, capmult=3.0, enableforce=false`가 있고 `verified:`를 통과할 것.
   (조건마다 `set_arm`이 확인하고 어긋나면 중단한다 — 기준 드라이버는 그 실패를 파이프로
   삼켰고 그 뒤 검사는 정책 이름만 봤다.)
2. **PolyServe arm의 tier 표가 `50:428,75:494,100:985`**일 것. `50:...,52:...`면 m1f를 읽은
   것이고 `25:...`면 m1을 읽은 것이다.
3. **PolyServe arm의 `placement_total{stage="forced"}`가 0**일 것. 0이 아니면
   `admission-binds`가 안 먹은 것이고 그 조건은 대조군과 같은 설정이다.
4. **llm-d arm은 조건마다 PRERUN 디렉토리를 하나 더 만든다.** 개수를 세는 모든 곳에서
   `grep -vc PRERUN`으로 거른다.
5. **엔진 4/4**. 파드의 startup probe는 포트 8000만 보므로 나머지 셋이 아직 가중치를 올리는
   중에도 Ready가 된다. `run_health.py`의 `eng` 열로 확인한다.
6. **거절률 0%인 arm에서는** `error_msg`를 종류별로 세고 "시작된 호출/초"가 trace의 도착률을
   넘는지 본다. 넘으면 그 구간은 정책이 아니라 부하 생성기를 잰 것이다.

### 5.2 예상과, 그것이 틀렸다고 말해 줄 값

| # | 예상 | 근거 | 반증 조건 |
|---|---|---|---|
| H1 | FluidServe가 네 arm 중 offered·admitted 둘 다에서 앞선다 | e2e 형태의 한 시간 trace에서 v0.4 후보가 llm-d를 앞섰고(§3의 기존 값), 정적에서도 EXP-68/80이 같은 방향이었다 | 어느 도착률에서든 다른 arm이 **반복 간 편차(1.4~3.3점)를 넘어** 앞서면 반증 |
| H2 | swe의 admitted가 네 arm 전부에서 e2e 형태보다 오른다 | 약속과 집행이 같은 양이 되면 승인이 지킬 수 있는 것만 받아들인다 | 어느 arm에서든 안 오르면, 그 arm에서는 형태가 원인이 아니라는 뜻이다 |
| H3 | **PolyServe가 m1f보다 t75에서 낫다** | m1f에서는 tier가 50/52/100이라 chat과 swe가 2 ms 차이여서 세 클래스를 셋으로 나눌 수 없었는데, t75에서는 **50/75/100 등간격**이 된다(사전 점검에서 확인) | 안 나아지면 PolyServe의 한계가 분류 축의 간격이 아니라 다른 데 있다는 것이고, EXP-106 §4의 서술을 고쳐야 한다 |
| H4 | PolyServe의 거부 사유에서 **memory가 여전히 지배적**이다 | EXP-106 smoke(m1f, 35 req/s, 4분)에서 거부 21,034건 중 memory가 84.6%, steady_state 15.1%, 시간 기반 셋의 나머지는 0.4% | steady_state가 memory를 넘으면, 예산이 넓어진 만큼 시간 검사가 다시 구속하기 시작했다는 것이고 그 자체가 결과다 |
| H5 | chat의 offered가 e2e 형태보다 내려간다 | 한 시간 trace에서 옛 30초 잣대로 재채점했을 때 chat offered가 11점 내려갔다(v0.4 §1.3) | 안 내려가면 그 대가가 한 시간 trace 특유의 것이지 형태의 성질이 아니다 |

### 5.3 점수는 어떻게 읽나

**넷을 같이 낸다** — offered 달성률(모든 도착이 분모), admitted 달성률, 거절률, token goodput.
그리고 요청 단위 지표만 보지 않는다: preemption 횟수, 엔진별 prefix hit rate,
`slo_rule_breakdown.py`의 TTFT/TBT/E2E 분해를 같이 본다.

**조건당 1회로 판정하지 않는다.** 반복 둘이 다 끝난 뒤에 판정하고, 반복 하나만 있는 시점의
수치는 "중간 관찰"로만 적는다.

**⚠ 이번 형태의 대가를 반드시 같이 적는다.** 한 시간 trace에서 옛 30초 잣대로 다시 채점하면
offered 79.2/79.0 → 70.3, admitted 96.8/96.5 → 92.2였고, admitted swe의 실제 e2e가 중앙값
34.2초에 30초 초과 62.7%였다. **정적 sweep에서도 같은 재채점을 해서 대가를 수치로 적는다.**

## 6. 이 실험이 답하지 않는 것

- **한 시간 동적 trace**는 범위 밖이다. 같은 형태로 다시 재야 하지만 별도 실험이다.
- **vLLM router와 Llumnix load-balance는 재실행하지 않는다.** 둘 다 SLO를 입력으로 받지
  않으므로 형태가 바뀌어도 결정이 같다. 기존 run을 `FS_SWE_TBT_MS=75`로 다시 채점하면 된다.
  **그 재채점은 아직 안 했다.**
- **PD-disaggregation**과 **엔진 쪽 dynamic chunking**은 여전히 범위 밖이다(엔진은 stock).
- **llm-d 안에서의 분해**(예측을 갖는 것 대 그 예측을 예산에 대고 쓰는 것)는 arm이 하나뿐이라
  못 한다.

## 7. 준비하면서 고친 것과 걸린 것

**7.1 llm-d 스택이 아예 없었다.** 2026-08-28 컨테이너 초기화로 `llmd` 네임스페이스와
InferencePool CRD가 사라졌고, 복구 절차(`RESTORE-2026-08-28.md` §3)가 llm-d를 언급하지 않는다.
`deploy/llmd/`의 매니페스트로 재구축했다 — CRD 둘, 예측기 셋, 라우터, 경로 B(InferencePool
`llmd-engines` + `sheddable`). ConfigMap 넷이 백업과 바이트 동일하고, `llmd-endpoints`만
엔진 파드 IP가 바뀌어 `llmd_endpoints.py`로 다시 만들었다(경로 B에서는 라우팅에 쓰이지 않고
마운트만 된다). 기능 확인: 라우터로 단발 요청 → 정상 응답, EPP 로그에 처리 기록.

**7.2 기준 드라이버 셋의 결함을 옮겨 심었다.** `run_exp27_mixsweep.sh` 계열에는
⑴ `set_scheduler_profiling.py`의 실패를 파이프가 삼키고 그 뒤 검사가 정책 이름만 보는 것,
⑵ arm 사이 환경 초기화가 없는 것, ⑶ v0.2 구성을 명시로 쓰지 않아 넷을 명시 안 한 arm이
컴파일 기본값을 조용히 얻는 것이 있다. 셋 다 이번 실험에서는 치명적이라(켜는 것이 전부
ablation 플래그다) `run_exp107_capforce.sh`에서 옮겨 왔다.

**7.3 llm-d 드라이버의 기본 도착률이 정본과 달랐다.** `900 1500 2100 2700 3000 3300 3600 4200`
(15·25·35·45·50·55·60·70 req/s)인데 정본 세트는 `600 900 1200 1500 2100 2700 3300 4200`이다.
정본의 llmdslo 열은 EXP-68(35~70)과 EXP-70/80(10~25)이 나눠 채운 것이라 드라이버 기본값이 그
목록이 아니었다. 스냅샷에서 정본 여덟 개로 바꿨다.

**7.4 첫 실행이 즉시 실패했다 — `unknown mix 't75'`.** 드라이버를 편집한 스크립트가 중간
assertion에서 멈추면서 `MIXCFG` 추가분이 파일에 써지기 전에 중단됐고, 다음 편집이 원본을 다시
읽어 덮어썼다. FluidServe arm이 30초 만에 실패했고 연쇄는 llm-d로 넘어갔다. **결과 디렉토리
0개, Job 0개로 오염은 없다.** 즉시 멈추고, 드라이버 자신의 `MIXCFG`·`ARMMIX`를 읽어 세 arm이
전부 실재하는 파일로 풀리는 것을 확인한 뒤 재시작했다.
→ **교훈**: 편집 스크립트가 여러 치환을 하면 **치환마다 파일에 쓰거나, 전부 성공한 뒤에만
쓰고 실패 시 원본을 그대로 둔다.** 이번 것은 후자였는데, 실패한 블록의 앞부분 치환이 성공했다는
출력만 보고 적용된 줄 알았다. **출력이 아니라 파일을 되읽어 확인해야 한다.**

**7.5 알려진 사소한 불일치**: `run_exp108_t75.sh`의 러너 Job 이름이 아직
`bench-runner-exp106-<arm>`이다(라벨만 exp108로 바꿨다). 실행 중인 스크립트는 편집하지 않으므로
sweep이 끝난 뒤에 고친다. llm-d 드라이버는 `bench-runner-exp108*`을 쓰므로 이름이 겹치지 않고,
연쇄는 Job 이름으로 대기하지 않는다.

## 8. 결과

*(반복 1이 끝나는 대로 채운다. 반복 하나짜리 수치는 중간 관찰로만 적고, 판정은 반복 둘이
끝난 뒤에 한다.)*
