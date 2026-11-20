# EXP-168: 섞인 논문 trace에서 PolyServe, Llumnix SLO(논문·변경), FluidServe v0.6 — 같은 세션

**날짜**: 2026-11-20 (실행 전 작성)

> **변경 (2026-11-20 10:10, 사용자 지시 "세션 다르다고 매번 fluidserve를 다시 돌리지마 제발 이미 충분히 돌렸어")**: v0.6(`fsg10dl`)을 뺐다. 체인은 이미 돌고 있어
> 드라이버(`run_exp168_hour.sh`)가 `fs*` arm을 받으면 바로 끝나게 했다(체인 로그에는 그 두 자리가 FAILED로 찍힌다). 6 run(기준선 셋 × 2회).
> v0.6과의 비교는 EXP-161(같은 trace, 모든 도착 80.69)을 쓰고 세션 간 문턱(모든 도착·거절률 max(4.6, 반복 범위), admitted max(1, 반복 범위))으로 판정한다.
**질문**: 세 클래스가 섞인 논문의 한 시간 trace(`dyn60_shift_m2Am1B_b1045`)에서 기준선을 지금 구성으로 다시 재고, FluidServe v0.6과 **같은 세션**에서 비교한다.
llm-d는 EXP-167에서 같은 trace로 쟀다(p90 모든 도착 56.76, admitted 98.71; 논문 값 재현). (2026-11-20 사용자 지시: "섞인 논문 trace에 polyserve, llumnix도 다시 해보자".)

## 구성

| arm | 정책 | 무엇이 논문 run과 다른가 |
|---|---|---|
| `polyservept75` | PolyServe(논문 구성: admission이 거절, elastic 파티션, KV admission 등 여섯 플래그) | tier 표가 run의 설정에서 만들어진다(`50:428,75:494,100:985`). 논문 한 시간 run의 드라이버는 기본 설정에서 표를 만들어 swe가 463토큰 기본값을 썼다(`evidence.md` §6 12번, EXP-165) |
| `slot75` | Llumnix SLO 논문 구성 | 없음 |
| `slot75inc` | Llumnix SLO + 처리 중 요청 기준 TPOT 판정(리비전에서 바꾼 것, `SLO_INCUMBENT_TPOT_GATE=1`) | 섞인 trace에서는 처리 중 요청의 budget이 새 요청보다 빡빡할 때 판정이 바뀐다. EXP-141(정적 35 req/s)에서는 admitted가 논문 구성보다 12.5점 낮았다 |
| `fsg10dl` | FluidServe v0.6 | — (EXP-161에서 같은 trace 모든 도착 80.69) |

- trace·설정: 기준선 셋은 `mix_dyn60_shift_m2Am1B_b1045_t75fair.json`(swe `tbt_ms` 75), v0.6은 `..._t75.json`(swe tier key 25, 실제 75 ms는 플래그) — `run_exp168_hour.sh`가 고르고 검사한다. 부하 생성 12개.
- 바이너리: 스케줄러 `7dc2355c`, 게이트웨이 `05745270`(EXP-164·165와 같다).
- 순서: r1 = PolyServe → Llumnix SLO → Llumnix SLO 변경 → v0.6, r2는 반대. 8 run, 약 9.6시간.
- 드라이버 `k8s/exp07/run_exp168_hour.sh`(= `run_exp165_hour.sh`, 이름만 바꿈), 체인 `/home/nxclab/scratch/exp168/run.sh`(run마다 `run_config.json` 확인, 스케줄러 지표 저장, 처리 중 요청 기준 판정 카운터 출력), 로그 `/home/nxclab/tools/exp168_chain.log`.

## 판정 (실행 전에 정함)

1. **같은 세션 비교**(v0.6 대 각 기준선, `slot75inc` 대 `slot75`): 아홉 지표 — 모든 도착, admitted, 거절률, 클래스별(chat·deepresearch·swe) admitted와 모든 도착 — 와 token goodput을 지표마다.
   문턱: 모든 도착·거절률·클래스별 모든 도착 4점, admitted·클래스별 admitted 1점, goodput 5 %, 또는 두 조건의 반복 범위.
2. **논문 값 대조**(EXP-109, ladder95, `evidence.md` §4.9; 논문 반복 범위 ± 세션 간 4.6점): PolyServe 모든 도착 52.7·거절 20.2 %, 클래스별 chat 49.5 / deepresearch 45.6 / swe 80.2;
   Llumnix SLO 27.9·거절 39.5 %, chat 30.0 / deepresearch 27.1 / swe 18.7. PolyServe는 tier 표가 다르므로 swe 쪽 차이는 그 때문일 수 있다. v0.6은 EXP-161(80.69)과 대조.
3. 같이 적는 것: 받은 요청의 마감 위반 종류(클래스별), TTFT·평균 토큰 간격, preemption, 엔진 KV 최대, PolyServe 거절 이유(memory / steady_state / 첫·둘째 토큰), 인스턴스별 클래스 구성,
   `slot75inc`의 `scheduler_slo_incumbent_gate_total{binding="resident"}`(0이면 판정이 한 번도 바뀌지 않은 것이므로 그 run은 `slot75`와 같은 것으로 본다).
4. **예상**: v0.6이 모든 도착·goodput에서 세 기준선보다 높다(EXP-161 80.7 대 논문 PolyServe 52.7·Llumnix SLO 27.9). PolyServe는 swe를 가장 잘 지키고(논문 swe 80.2) chat을 받은 뒤 놓친다.
   `slot75inc`는 `slot75`보다 admitted가 낮거나 같다(EXP-141과 같은 방향).

## 결과 (2026-11-20 10:05 ~ 17:29 KST, 기준선 6 run, ladder95)

체인 `=== EXP168 CHAIN DONE 2026-11-20 17:29:33`(`bin/` 복원, 게이트웨이 replica 0, Sionna 복귀). v0.6 두 자리는 의도대로 건너뜀(로그의 FAILED·WARN 두 줄씩). 6 run 모두 워크로드 확인(swe 7000/75 fair 설정),
PolyServe tier 표 `50:428,75:494,100:985` 확인, `slot75inc`의 처리 중 요청 기준 판정 r1 815,308·r2 819,382회(자기 budget 410,512·384,370회), `slot75`는 카운터 없음(판정 꺼짐).
⚠ 건너뛴 v0.6 자리 뒤에 저장된 `metrics_<rep>_fsg10dl_shift.txt`는 그때 떠 있던 `slot75inc` 스케줄러의 값이므로 쓰지 않는다. 모든 run이 trace의 도착을 전부 냈고 거절이 아닌 오류 0. 집계 `/home/nxclab/scratch/exp168/obs_*.txt`.

| arm (2회 평균, 반복 범위) | 모든 도착 | admitted | 거절률 | chat 모든 도착 / admitted | deepresearch 모든 도착 / admitted | swe 모든 도착 / admitted | goodput (tok/s) | preemption r1·r2 | KV 평균 / p90 | 엔진 토큰 간격 |
|---|---|---|---|---|---|---|---|---|---|---|
| PolyServe | 53.70 (0.67) | 65.58 | 17.9 % | 51.85 / 54.75 | 42.96 / 99.17 | 80.66 / 99.36 | 8,653 | 0 · 0 | 0.44 / 0.60 | 56.2 ms |
| Llumnix SLO (논문) | 27.69 (0.00) | 45.97 | 39.5 % | 29.85 / 55.20 | 26.91 / 29.60 | 18.14 / 44.70 | 6,786 | 6,054 · 6,091 | 0.76 / 1.00 | 68.0 ms |
| Llumnix SLO (처리 중 요청 기준) | 43.58 (0.10) | 51.62 | 15.3 % | 37.48 / 38.65 | 63.73 / 99.68 | 40.97 / 74.14 (범위 6.4) | 7,773 | 6,136 · 6,793 | 0.73 / 1.00 | 72.6 ms |
| (참고) FluidServe v0.6, EXP-161 | 80.69 | 99.90 | 19.0 % | 85.43 / 99.99 | 69.19 / 99.86 | 75.89 / 99.41 | 11,764 | 1,612 · 1,728 | 0.57 / 0.99 | 55.6 ms |
| (참고) llm-d p90, EXP-167 | 56.76 | 98.70 | 42.4 % | 55.73 / 98.85 | 66.42 / 99.58 | 45.96 / 95.72 | 8,950 | 85 · 166 | 0.38 / 0.52 | 42.6 ms |
| (참고) llm-d p50, EXP-167 | 63.07 | 94.16 | 32.9 % | 67.83 / 92.34 | 54.83 / 99.59 | 52.79 / 97.40 | 9,396 | 52 · 54 | 0.44 / 0.61 | 48.7 ms |

PolyServe 거절 판정(r2, 후보·재시도마다): memory 156,735(자기 tier) + 238,191(promotion), steady_state 4,000 + 44,165, 첫·둘째 토큰 390. tier 배분 chat 2 / swe 1 / deepresearch 1(r1·r2 같다).

### 판정

**Llumnix SLO 처리 중 요청 기준 대 논문 구성 (같은 세션, 문턱 4/1점, goodput 5 % 또는 반복 범위)**: 열 지표 중 아홉이 "다르다(좋음)" — 모든 도착 +15.9, admitted +5.7, 거절률 −24.2, chat 모든 도착 +7.6,
deepresearch 모든 도착 +36.8·admitted +70.1, swe 모든 도착 +22.8·admitted +29.4(문턱 6.4), goodput +14.5 %. **chat admitted만 −16.6으로 나쁨**(55.2 → 38.7).
EXP-141(정적 35 req/s)에서 admitted가 12.5점 낮았던 것과 반대 방향이다 — 그쪽은 정적 부하에서 쟀다.

**각 기준선 대 v0.6 (EXP-161, 세션이 다르다; 문턱 모든 도착·거절률·클래스별 모든 도착 max(4.6, 범위), admitted max(1, 범위), goodput max(5 %, 범위))**:
- PolyServe: 모든 도착 −27.0, admitted −34.3, chat 모든 도착 −33.6·admitted −45.2, deepresearch 모든 도착 −26.2, goodput −26 % **나쁨**; 거절률·deepresearch admitted·swe admitted 차이 없음; **swe 모든 도착 +4.8로 PolyServe가 좋음**(문턱 4.6 바로 위; 논문의 "PolyServe performs best for the Agent workload"와 같은 방향).
- Llumnix SLO 논문: 열 지표 모두 나쁨(모든 도착 −53.0, goodput −42 %).
- Llumnix SLO 처리 중 요청 기준: 거절률·deepresearch admitted 차이 없음, 나머지 여덟 나쁨(모든 도착 −37.1, goodput −34 %).

**논문 값 재현 (EXP-109, `evidence.md` §4.9; 논문 반복 범위 ± 세션 간 4.6점)**: PolyServe 53.70 대 52.7, 거절 17.9 대 20.2 %, chat 51.85 대 49.5, deepresearch 42.96 대 45.6, swe 80.66 대 80.2 — **재현**
(이번은 tier 표가 swe 494토큰, 논문 run은 드라이버가 463토큰 기본값을 썼을 가능성 — swe 차이 0.5점). Llumnix SLO 27.69 대 27.9, 거절 39.45 대 39.5 %, chat 29.85 대 30.0, deepresearch 26.91 대 27.1, swe 18.14 대 18.7 — **재현**.

### 해석 (원인은 EXP-169 §5)

- PolyServe: chat 엔진 둘은 정상상태 예측이 prefill을 빼서 실측의 0.59~0.70이라 budget을 넘기고(받은 chat 55 % 통과), swe·deepresearch 엔진 둘은 메모리 판정(논리 토큰, promotion에서는 받으려는 요청의 기대 출력 × chat batch)으로 KV 0.3~0.6에서 멈춘다. tier 배분은 한 시간 내내 2/1/1.
- Llumnix SLO 논문: deepresearch가 KV를 1.00까지 채워 preemption 6천 회, 받은 deepresearch의 30 %만 통과. 처리 중 요청 기준은 모든 클래스를 예측 TPOT ≤ 45 ms에서 받게 해 deepresearch가 KV를 혼자 채우지 못하지만(admitted 99.7), 예측이 실측의 0.46~0.62라 chat이 평균 간격으로 실패한다(chat admitted 38.7).
- 섞인 trace에서 v0.6은 모든 기준선보다 모든 도착·goodput이 높다(goodput 11,764 대 6,786~9,396). PolyServe가 swe 모든 도착에서만 4.8점 앞선다.
