# EXP-166: llm-d의 예측기를 90번째 백분위수 대신 중앙값으로 — chat 2.5배

**날짜**: 2026-11-19 (실행 전 작성)
**질문**: 논문 구성의 llm-d가 chat에서 FluidServe v0.6보다 더 거절하는 것(EXP-165 반복 1: 거절률 38.9 % 대 31.7 %, token goodput 17,973 대 19,862 tok/s)이
llm-d 예측기의 여유 때문인가. llm-d의 예측기는 TTFT·TPOT의 **90번째 백분위수**를 학습하고(`LATENCY_QUANTILE_ALPHA` 기본값 0.9, xgboost `quantile_alpha`),
EPP는 그 예측을 요청의 SLO와 여유 없이 비교한다(`sloBufferFactor` 1; `prediction.go:149-177`의 `pred.TPOT < tpotSLO`). 그래서 "열에 아홉은 SLO 안"일 때만 가능으로 본다.
EXP-165에서 EPP 예측 평균은 chat TTFT 313 대 실측 238 ms, TPOT 47.7 대 44.2 ms였고, 받은 chat의 평균 토큰 간격 중앙값은 44.3 ms로 v0.5(판정선 0.9배, 44.2 ms)와 같고
v0.6(1.0배, 49.1 ms)보다 낮았다. 중앙값(p50)을 학습시키면 v0.6처럼 "중간 예측을 budget 그대로와 비교"하는 쪽이 된다. (2026-11-19 사용자 결정 "좋아 해보자":
논문 구성 p90은 기준선으로 두고 p50을 민감도 변형으로 추가, chat 2.5배부터.)

## 구성

| 항목 | 값 |
|---|---|
| 정책 | llm-d, EXP-165와 같은 논문 버전(EPP llm-d-router `644a8856` 다시 빌드, 예측기 `ec7626b`) + **`LATENCY_QUANTILE_ALPHA=0.5`** (training·prediction ConfigMap 둘 다; `deploy/llmd/predictor-paper-p50.yaml`). arm 이름 `llmdp50slot75` |
| 비교 대상 | EXP-165의 llm-d p90(`llmdslot75`, `exp165mr1`·`exp165mr2`의 `onlychat25`) — 같은 날, 바로 앞 체인 |
| trace·설정 | `onlychat25`: `dyn60_onlychat_b25112.csv`, `mix_dyn60_onlychat_b25112_t75fair.json`, 부하 생성 36개 |
| 반복 | 2회 |
| 드라이버·체인 | `k8s/exp07/run_exp166_llmd.sh`(= `run_exp165_llmd.sh` + preflight에서 두 ConfigMap의 0.5 확인, 10분 뒤 학습 로그의 `for quantile 0.5` 확인; 그 밖의 검사는 같다: 이미지·바이너리·EPP 커밋, 표본 0개 시작, 학습 성공, EPP 예측이 10 ms 고정이 아님).
  체인 `/home/nxclab/scratch/exp166/run.sh`(EXP-165 llm-d 체인이 끝난 뒤 p50 manifest 적용 → 2 run → `predictor-paper.yaml`로 되돌리고 키가 사라졌는지 확인 → Sionna), 로그 `/home/nxclab/tools/exp166_chain.log` |

## 판정 (실행 전에 정함)

1. 지표: 모든 도착, admitted, 거절률, token goodput. "평균 차이 > max(문턱, 두 조건 각각의 반복 범위)"이면 "다르다". 문턱은 모든 도착·거절률 4점, admitted 1점, goodput은 같은 규칙을 상대값 5 %로.
   (EXP-165의 p90과 이 실험의 p50은 같은 날 이어진 체인이지만 엔진·예측기를 run마다 다시 시작하므로 같은 세션으로 본다.)
2. 같이 적는 것: 받은 chat의 마감 위반 종류, 평균 토큰 간격 p50/p90과 budget 초과 비율, TTFT p99, EPP 예측 평균 대 실측(TTFT·TPOT), 5분 거절률, 엔진별 요청 비율.
3. **예상**: 거절률이 내려가고(모든 도착·goodput 상승), 받은 chat의 평균 토큰 간격이 budget 50 ms에 붙어 평균 간격 위반이 늘어 admitted가 내려간다(판정선 0.9 → 1.0이었던 v0.5 → v0.6과 같은 모양;
   v0.6은 첫 토큰 마감 판정과 iteration 예측의 정확도로 admitted를 지켰다). 예측이 TPOT에서 실측보다 약 8 % 높았으므로(p90) p50의 예측은 실측과 비슷하거나 낮아질 것이다.
4. 해석: 모든 도착이 v0.6(68.16)에 가까워지고 admitted가 99 근처에 머물면 "llm-d가 chat에서 덜 받은 것은 p90 여유 때문"이고, admitted가 크게 내려가면 "p90 여유가 llm-d의 admitted를 지키던 것"이다.
   swe만·deepresearch만에서 llm-d의 실패(몰림 위반, 엔진 두 대 쏠림과 preemption)는 예측의 여유와 무관하므로 이 실험의 대상이 아니다.
