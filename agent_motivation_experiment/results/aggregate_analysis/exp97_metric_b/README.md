# EXP-97 그림 — 클래스 선호의 정규화를 고쳤을 때 한 시간 믹스 이동 trace에서 무엇이 달라지나

**판정과 표의 정본은 `experiments/EXP-97_normalisation-on-the-hour.md` §5·§6.**
여기는 그림마다 **무엇을 주장하는지 / 무엇을 말하지 않는지**만 적는다.

두 디렉토리가 두 반복이다. **한쪽만 보고 판정하지 않는다** — 이 실험에서 고친 방식의
반복 폭이 총계 4.2점, 마지막 구간 11.5점이다.

| 디렉토리 | 반복 |
|---|---|
| `exp97_metric/` | 반복 1 (`exp97r1_*`, 선호 끔은 `exp93nr1`) |
| `exp97_metric_b/` | 반복 2 (`exp97br1_*`, 선호 끔은 `exp93nbr1`) |

## arm 셋을 부르는 이름

| 그림의 범례 | 문서에서 부르는 이름 | 정렬이 무엇으로 순위를 매기나 | 디렉토리 |
|---|---|---|---|
| `preference off` | **선호 끔** | 클래스 항이 없다 (`room`만) | `fsnoaff` |
| `preference, shipped` | **지금 방식** | 그 인스턴스 안에서 같은 클래스가 차지하는 **비율** | `fspfx` |
| `preference, fixed` | **고친 방식** | 그 인스턴스가 든 같은 클래스의 **개수** | `fscount` |

## 그림

| 파일 | 무엇을 주장하나 | 말하지 않는 것 |
|---|---|---|
| `trace_shape.png` | 세 arm이 같은 도착 과정과 같은 믹스 일정을 받았다 | arm별 차이는 여기 없다 |
| `exp97_classpreference_normalisation_shift_timeline.png` | **주 그림.** 8패널. C·D가 두 분모의 달성률, E·F가 클래스별, H가 함대 KV | 90초 창이므로 그보다 짧은 사건은 안 보인다 |
| `class_goodput_hour.png` | **손실의 위치.** 고친 방식의 chat이 **약 53~58분에만** 5,000 → 500 tok/s로 무너지고 나머지는 겹친다 | 어느 인스턴스에서 무너졌는지는 엔진 그림에서 본다 |
| `compare_rpm_hour.png` | 세 arm을 열로 놓고 배치·큐·KV·디코드를 행으로 | 행마다 y축이 열 사이에 공유된다 — 열 하나만 보고 "한가하다"고 읽으면 안 된다 |
| `exp97_shift_engine_timeline.png` | 엔진 넷을 따로 그린 시계열 | |
| `exp97_shift_engine_requests.png` | 엔진별 요청 수·클래스 구성·chat 토큰당 시간 | 스케줄러 로그 조인이므로 부하 최대 구간에서 줄이 빠질 수 있다 |
| `engine_*` / `tokens_*` / `llumnix_*` | run 하나씩의 엔진 계층·토큰 처리량·컨트롤플레인 | 전부 엔진과 게이트웨이 자신의 Prometheus 값이라 클라이언트 지표 결함에 영향받지 않는다 |
| `separation_hour.csv` | 클래스당 유효 인스턴스 수, 이동량, 최다 보유 교체 횟수, chat 없는 인스턴스 비율 | |

## 그림에서 읽어야 하는 것 셋

**① 정렬 수정은 작동한다** — 최다 보유 인스턴스 교체가 chat에서 시간당 58.3/53.4(선호 끔) →
17.8/7.9(지금 방식) → **7.9/5.9(고친 방식)**이고, chat을 든 적 있는 인스턴스가 4 → 3 → **2**곳이다.

**② 그 대가는 메모리다** — 주 그림 패널 H에서 고친 방식의 함대 KV가 48~60분에 **70~77.5%**이고
지금 방식은 60~65%다. 같은 구간에서 패널 B의 거절률이 50%까지 오르고 패널 D의 admitted
달성률이 40%로 떨어진다 — **더 거절하고도 받아들인 것을 못 지켰다.**

**③ 그 구간이 trace의 최고 부하다** — 패널 A에서 48~60분이 35~46 req/s다.

## ⚠ chat 없는 인스턴스 비율의 두 열은 다른 양이다

`separation_hour.csv`의 `nochat_pct`(클라이언트 기록)와 `gate_nochat_pct`(스케줄러 게이지)가
**반복 1의 고친 방식에서 12.4 대 42.5로 30점 벌어지고 반복 2에서는 36.2 대 36.6으로 일치한다.**

클라이언트 기반은 요청이 dispatch된 인스턴스에 시작부터 종료까지 상주한다고 보고, 스케줄러
게이지는 **남은 예산으로 더 이상 달성할 수 없게 된 요청을 게이트에서 뺀다.** 반복 1이 chat
토큰당 초과가 41.9%였던 run이고 반복 2는 18.9%였으므로, **불일치의 크기가 예산 초과의 크기를
따라간다. 결함이 아니라 이 실험이 재려던 기제의 결과다.**

## 다시 만들려면

```bash
cd Agent_applications/agent_motivation_experiment
bash analysis_scripts/redraw_hour_trace_exp97.sh results/aggregate_analysis/exp97_metric      # 반복 1
bash analysis_scripts/redraw_hour_trace_exp97.sh results/aggregate_analysis/exp97_metric_b b  # 반복 2
```

⚠ **`fsnoaff`를 `exp38_policy_compare.py`의 `ARM_ORDER`·`ARM_LABEL`에 등록해야 한다** —
2026-08-25에 등록했다. 등록 전에는 그 스크립트가 종료하며 이유를 찍는다(조용히 빠지지 않는다).
그리고 그 arm의 세션 태그는 반복 1이 `exp93nr1`, 반복 2가 `exp93nbr1`이라 `exp93r1`로 찾으면
빈 결과가 나온다.
