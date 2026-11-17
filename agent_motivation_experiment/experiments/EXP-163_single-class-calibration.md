# EXP-163: 클래스 하나만 있는 한 시간 trace — v0.6(후보4)으로 도착률·prefix 적중 보정

**날짜**: 2026-11-17 (실행 전 작성)
**목적**: 한 번도 재지 않은 단일 SLO 조건(chat만 / swe만 / deepresearch만)을 연다. 클래스가 하나면 클래스 배분·분리가 생기지 않으므로,
예측기(토큰당 시간, 첫 토큰, KV)와 horizon이 admission에 어떻게 작용하는지만 따로 볼 수 있다(2026-11-17 사용자 결정: horizon 재검토를 이 조건에서 시작).
이 실험은 **보정**이다 — v0.6 하나로 클래스마다 1회 돌려 적절한 도착률과 prefix 적중률을 본 뒤, 도착률을 조절해 v0.5와 v0.6을 함께 돌린다(사용자 결정).

## 구성

| 항목 | 값 |
|---|---|
| 구성 | `fsg10dl` = v0.6(채택 예정, 아직 후보4 이름) = v0.5 + 판정선 1.0 × budget + 첫 토큰 마감 판정. 스케줄러 `7dc2355c`, 게이트웨이 `05745270`, h100 |
| trace | 지금 한 시간 trace와 도착 시각이 같고(arrival_s md5 `5256bef0…`) 모든 도착을 한 클래스로 바꾼 셋(`traces/dynamic/relabel_fixed_mix.py`): `dyn60_onlychat_b1045.csv`, `dyn60_onlyswe_b1045.csv`, `dyn60_onlydr_b1045.csv` — 각 99,242건, 초당 10~45건 |
| 워크로드 설정 | `mix_dyn60_onlychat_b1045_t75.json`, `mix_dyn60_onlyswe_b1045_t75r.json`(swe 반복 번호 켬 — 1,500개 기록이 평균 약 54 s, 가장 바쁜 분 약 33 s마다 다시 온다), `mix_dyn60_onlydr_b1045_t75.json`. budget은 t75 형태(chat 5 s/50 ms, deepresearch 10 s/100 ms, swe 7 s/75 ms) |
| 순서 | chat만 → swe만 → deepresearch만, 각 1회. 약 3.8시간. EXP-162 체인이 끝나면 시작 |
| 드라이버·체인 | `k8s/exp07/run_exp163_hour.sh`(변형 `onlychat`·`onlyswe`·`onlydr`, 변형마다 맞는 설정이 아니면 멈춤 — 시험 완료), `/home/nxclab/scratch/exp163/run.sh`(run마다 클래스 계획 파일·swe ttft 7000·반복 번호를 확인, 아니면 WARN), 로그 `/home/nxclab/tools/exp163_chain.log` |

데이터 반복: chat은 대화 10,000개를 차례 단위로 펼쳐 순서대로 돌리고, deepresearch는 60,000건이라 한 시간 99,242건이면 약 36분 뒤부터 같은 질문이 다시 나온다. 둘 다 반복 간격이 엔진의 prompt 유지 시간(약 1분)보다 길다 — prefix 적중률로 확인한다.

## 이번에 보는 것 (판정이 아니라 다음 실험의 도착률을 정하기 위한 관찰)

- 클래스마다: 모든 도착·admitted·거절률, 분 단위 거절률(부하가 admission을 움직이는 구간이 있는가), 받은 요청의 마감 위반 종류, TTFT p50/p90/p99, preemption, goodput.
- prefix 적중: 결정 로그의 `prefillraw / prompt`(거의 통째 적중 비율, 적중 토큰 중앙값), 엔진 보고 prefix 적중률.
- 예측기: 예측 iteration 시간 대 실측(gauge), 예측 첫 토큰(결정 로그의 waited + prefillest) 대 실제 TTFT, KV 투영 대 실제.
- 운영: 분당 서로 다른 요청 수 대 trace(클라이언트 재전송 여부), 거절이 아닌 오류, 결정 지연.
- **도착률 조절의 기준(안)**: 거절률이 한 시간 평균 20~40 % 안팎이고, 부하가 낮은 구간에서는 거의 받고 높은 구간에서는 admission이 결정을 바꾸는 조건. 정하는 것은 결과를 보고 사용자와 함께.

## 결과

(실행 후 적는다)
