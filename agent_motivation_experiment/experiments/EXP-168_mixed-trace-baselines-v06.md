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
