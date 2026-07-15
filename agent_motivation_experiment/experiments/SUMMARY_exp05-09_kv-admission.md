# SUMMARY — EXP-05→09: KV-occupancy admission 아크 (인사이트 · 방법 · 세팅)

**기간**: 2026-07-08 ~ 07-15 · **스택**: NXC13 8×B200, Llumnix full-mode
(gateway :8089 / scheduler :8088 / 4×TP2 vLLM), 상세 문서는 각 EXP 링크.

## 스토리라인 (한 단락)

무제어 서빙이 chat/agent 양쪽에서 congestion collapse를 보이는 것을 정량화
(EXP-04/05/06) → KV를 "유량으로 채워지는 수조"로 보는 가설을 세우고 유량
분석으로 검증(ANALYSIS_kv-tank-flow) → 대조군으로 "가장 단순한" 순간-점유율
임계 admission을 구현해 돌렸더니 예상을 뒤엎고 정상상태에서 거의 최적
(EXP-07/08) → 왜인지 파고들어 **TBT가 hot KV 총량에 선형**이라는 법칙을 발견
(ANALYSIS_why-not-full-kv) → agent 체인 워크로드로 전이 확인 + request-level
admission의 job-level 비용과 SLO-정의 의존성을 실측(EXP-09).

## 핵심 인사이트 5개

1. **무제어 붕괴는 극적이고 2단계다.** chat: 60→80 req/s에서 attain 77→3%
   (EXP-04), 90 req/s에서 1.5%(EXP-07 off). SWE: TBT 위반이 먼저(KV ~65%),
   이어 큐잉이 TTFT까지 무너뜨림(EXP-06). 토큰 처리량은 그동안 멀쩡해 보임.
2. **TBT ≈ 10 + 20.6 × hot-KV[Mtok] ms — 워크로드 불변.** chat(840 tok/req)과
   SWE(5.5k tok/req)가 batch 수 축에선 7.5× 다르지만 KV 축에선 한 곡선
   (bin별 1–2ms 일치). decode step 시간은 attention이 매 step 읽는 KV 총량이
   결정(bandwidth bound). → 50ms SLO = KV 예산 ~83%(중앙값 상한); 실전 θ\*는
   꼬리/요동/prefill/90%+ 불안정 마진으로 그 아래. **점유율 신호가 준-인과라
   occupancy thresholding이 잘 작동하는 이유이자, "KV를 다 쓰지 않는 게
   최적"인 이유.** (ANALYSIS_why-not-full-kv)
3. **θ\* = 0.6 (meanTBT≤50ms 기준, chat·SWE 공통).** chat: offered 곡선이
   용량 클램프 min(1, 60/rate)에 전 구간 밀착, feasible 구간 무손실, admitted
   96–100% (EXP-08). SWE: admitted 99.4–100% 수평, offered 2–3× vs off
   (EXP-09). cross-run 재현 편차 ≤2%p.
4. **request-level admission의 남은 결함 4개** (EXP-07 §4 + EXP-09):
   (a) θ는 오프라인 sweep 산물 — 자동 동작점 탐색 없음;
   (b) **job-무지**: mid-chain 거절 1건 = 체인 사망 + 후속 콜 4.6× 억압
   (완주율은 무제어보다 오히려 낮음);
   (c) **SLO-정의 의존**: p95 ITL≤50ms 기준이면 θ\*=0.3–0.4 (mean이 per-token
   꼬리를 흡수 — ITL CDF에서 θ0.6의 p95는 170–200ms);
   (d) 300–700ms prefill-stall 꼬리는 θ로 제거 불가(chunked-prefill 스케줄링).
5. **수조(tank) 관점은 예보와 setpoint를 준다.** 순유량 외삽으로 핀 ~60s 전
   예보(G1), 배수 곡선의 churn 경계(G2), hot/idle 2층 구조와 hit-침식 되먹임
   (87.6→74% → prefill 수요 ~2×). 인사이트 2와 결합하면 setpoint를
   `(SLO−절편)/기울기`로 계산 가능 — 튜닝 없는 θ 예측 가설.

## 실험 방식 (모든 EXP 공통 방법론)

| 항목 | 정의 |
|---|---|
| SLO | TTFT ≤ 5s (도착 기준, gateway 큐 포함) AND per-request meanTBT ≤ 50ms |
| steady window | [60s, dur−20s], 도착(start_time) 기준; 60s sliding/10s step 병용 |
| 제외 범주 | error/timeout/run-end-cut 제외·별도 보고; **rejected는 별도 범주** — offered 뷰에서 위반으로 카운트, admitted 뷰에서 제외 |
| job goodput | 제출창 [60,360)s job만 분류(우측 절단 방지); 완주 / 완주+전콜SLO 두 변형 |
| 조건 독립성 | 조건마다 engine+control-plane cold restart (`--restart-per-condition`) |
| warmup | chat 20 req/s×60s, SWE 0.25 jobs/s×60s (EXP-04의 herd 교훈) |

## 세팅 요약

| EXP | 워크로드 | 모델 | rate | θ | 조건 |
|---|---|---|---|---|---|
| [04](EXP-04_8192conc-chat-sweep.md) | chat(sharegpt) | 3-70B | 5–120 req/s | – | 8 |
| [05](EXP-05_warmup-chat-sweep.md) | chat + warmup | 3-70B | 5–100 req/s | – | 11 |
| [06](EXP-06_swe-tool-delay-sweep.md) | SWE chain(4–12콜, tool-delay) | 3.1-70B/40960 | 0.25–5 jobs/s | – | 11 |
| [07](EXP-07_kv-threshold-admission.md) | chat | 3.1-70B | 50/60/90 req/s | 0.3–0.8+off | 12 |
| [08](EXP-08_kv-threshold-full-sweep.md) | chat | 3.1-70B | 5–100 req/s | 0.6 | 11 |
| [09](EXP-09_swe-kv-admission-sweep.md) | SWE chain | 3.1-70B | 0.25–5 jobs/s | 0.3/0.4/0.5/0.6 | 44 |

공통: 4×TP2 util 0.9, chunked prefill 8192, prefix caching ON, 8 procs ×
1024 threads(=8192 conc), in-cluster runner. 측정 5 min/조건(chat),
12 min(SWE).

## 구현 (admission 장치)

- llumnix `feat/kv-admission-threshold`: scheduler 새 메트릭
  `kv_cache_usage_ratio`(hot 점유율) + `--admission-kv-usage-threshold θ`
  per-instance 하드 필터(전 인스턴스 ≥θ → 429), gateway
  `--wait-scheduling-timeout=0s`로 즉시 503. CGO 없는 정적 바이너리를
  hostPath로 stock 이미지에 주입 (`k8s/exp07/patch-scheduler-kvadm.sh`).
- Agent_applications `feat/exp07-kv-admission`: 클라 503→`is_rejected/
  KV_THRESHOLD`, 분석 rejected 범주, 드라이버
  (`k8s/exp07/run_exp07.sh`, `run_exp09_swe.sh`).

## 그림/스크립트 포인터

`results/aggregate_analysis/{exp0N}/…` + 생성 스크립트
`analysis_scripts/request_level/plot_exp0{7,8,9}_*.py`,
`analyze_tbt_drivers.py`(TBT–KV 법칙), `slo_sliding_window.py`,
`plot_slo_vs_throughput.py`, `job_level/job_goodput_sweep.py`.

## 열린 질문 / 다음 후보

- flow-기반 컨트롤러의 실증 여지: 튜닝-무관 setpoint 계산, burst/transient
  (정상상태에선 θ가 이미 근사-최적이므로 차별화 무대는 비정상 상태).
- job-aware admission: 체인 첫 콜에서만 거절(mid-chain 보호) — 인사이트 4(b)
  의 직접 해법 후보.
- EXP-10(open-loop request replay) 설계 완료·보류 (EXP-09 문서 말미).
- vLLM 카운터 패치로 유출 분해(완료/preempt/evict), per-step ITL 계측으로
  TBT–KV 기울기 직접 검증.
