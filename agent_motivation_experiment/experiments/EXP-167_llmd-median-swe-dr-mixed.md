# EXP-167: llm-d 중앙값 예측기 — swe만·deepresearch만, 그리고 세 클래스가 섞인 논문 trace에서 p90 대 p50

**날짜**: 2026-11-19 (실행 전 작성)
**질문**: EXP-166(chat 2.5배)에서 보는 "llm-d 예측기의 90번째 백분위수 여유"가 swe만·deepresearch만과 세 클래스가 섞인 trace에서는 어떻게 작용하는가.
llm-d의 예측기 백분위수(`LATENCY_QUANTILE_ALPHA`)는 클래스마다 따로 정하는 값이 아니라 예측기 전체에 하나다 — 섞인 trace에서는 chat·swe·deepresearch 요청이 같은 모델로 예측되고,
요청마다 다른 것은 헤더의 SLO(5000/50, 7000/75, 10000/100)뿐이다. (2026-11-19 사용자 지시: "swe랑 deepresearch도 하고, 그리고 세개 섞인것도 해봐야해".)

## 구성

| 항목 | 값 |
|---|---|
| 정책 | llm-d 논문 버전(EPP `644a8856` 다시 빌드, 예측기 `ec7626b`). p90 = 논문 구성(`deploy/llmd/predictor-paper.yaml`, 키 없음 = 예측기 기본값 0.9, arm `llmdslot75`), p50 = `predictor-paper-p50.yaml`(arm `llmdp50slot75`) |
| run (반복마다) | swe만 p50, deepresearch만 p50, 섞인 trace p90·p50(r1은 p90 → p50, r2는 p50 → p90). 2회, 8 run, 약 9.5시간 |
| 섞인 trace | 논문의 한 시간 trace `dyn60_shift_m2Am1B_b1045.csv` + `mix_dyn60_shift_m2Am1B_b1045_t75fair.json`, 부하 생성 12개 — 논문 llm-d run(`run_exp109_llmd.sh` `full`)과 같은 쌍 |
| swe만·deepresearch만 | EXP-165와 같은 trace·fair 설정(`..._onlyswe_b1045_t75rfair.json`, `..._onlydr_b1045_t75fair.json`), 부하 생성 12개 |
| 드라이버 | `k8s/exp07/run_exp167_llmd.sh` = `run_exp166_llmd.sh` + `LLMD_Q`(0.5/0.9)로 백분위수를 run마다 고르고 검사(0.5는 두 ConfigMap 모두 0.5, 0.9는 두 ConfigMap 모두 키 없음; 10분 뒤 학습 로그의 `for quantile <값>`), 변형 `shift` 추가. 지금 p50 상태에서 0.5 통과·0.9 ABORT 확인 |
| 체인 | `/home/nxclab/scratch/exp167/run.sh`(EXP-166 체인 뒤; run마다 그 백분위수의 manifest를 적용 → 드라이버가 예측기를 재시작하며 반영; 끝에 p90으로 되돌리고 키가 없는지 확인 → Sionna), 로그 `/home/nxclab/tools/exp167_chain.log` |

## 판정 (실행 전에 정함)

1. **swe만·deepresearch만 p50 대 p90**: p90은 EXP-165 llm-d(`exp165mr1`·`exp165mr2`, 같은 날 앞 체인). 지표 모든 도착, admitted, 거절률, token goodput. "평균 차이 > max(문턱, 두 조건 각각의 반복 범위)"이면 "다르다";
   문턱 모든 도착·거절률 4점, admitted 1점, goodput 5 %. EXP-165에서 llm-d 자신의 반복 범위가 이미 컸으므로(swe admitted 7.8점, deepresearch 모든 도착 6.5점) 그 범위가 문턱이 될 수 있다.
2. **섞인 trace p50 대 p90**(같은 체인, 순서 교차): 아홉 지표 — 모든 도착, admitted, 거절률, 클래스별(chat·deepresearch·swe) admitted와 클래스별 모든 도착 — 와 token goodput을 지표마다 따로.
   문턱은 모든 도착·거절률·클래스별 모든 도착 4점, admitted·클래스별 admitted 1점, goodput 5 %, 또는 반복 범위.
3. **재현 확인(p90, 섞인 trace)**: 논문 값은 EXP-109 2회 평균 모든 도착 53.0(반복 49.9·56.3), 거절률 45.7 %(49.0·42.4), 클래스별 모든 도착 chat 52.9·deepresearch 56.5·swe 47.8(`evidence.md` §4.9).
   이번 p90 두 run의 평균이 그 반복 범위 ± 세션 간 편차(4.6점) 안이면 "논문 llm-d를 재현했다"고 적는다. 벗어나면 원인(예측기 버전 추정, 바이너리 재빌드, 클러스터 재구축)을 따진다.
4. 같이 적는 것: 받은 요청의 마감 위반 종류, TTFT·평균 토큰 간격 분포, EPP 예측 평균 대 실측(TTFT·TPOT), 엔진별 요청 비율·KV 최대, preemption, 5분 거절률, 거절 분류(`is_rejected`), 클라이언트 상태.
   섞인 trace에서는 인스턴스별 클래스 구성(논문 그림 4: llm-d가 chat을 네 대에 퍼뜨린다)도 본다.
5. **예상**: swe만·deepresearch만은 p50이 더 받아 거절률이 내려가지만, 이 trace들에서 llm-d의 실패는 몰림 위반과 KV 초과·preemption이므로 admitted가 더 내려간다.
   섞인 trace는 chat이 대부분이므로 chat 2.5배(EXP-166)와 같은 방향(더 받고 admitted는 조금 내려감)이고, swe·deepresearch의 admitted는 내려간다.
