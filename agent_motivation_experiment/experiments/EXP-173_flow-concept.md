# EXP-173: KV와 첫 토큰을 하나의 유량 개념으로 — 후보를 하나씩 검증

**날짜**: 2026-11-22 (실행 전 작성; 후보 arm은 각자 실행 전에 판정 규칙을 덧붙인다)
**사용자 지시 (2026-11-22 01:00)**: "KV cache는 유량 분석을 하듯이 계산해야 한다 — 들어오는 양과 나가는 양을 보면서 들어오는 밸브를 조절해야 하고, 그러려면 들어올 양도 생각해야 한다. 첫 토큰 판정의 과대 예측도 문제이고 Little의 법칙이 그 관점에서 맞는 것 같다. 하나의 일관된 컨셉으로 prefill 대기 시간 문제와 KV 문제를 잡을 수 있지 않을까." "여러 안을 만들어 밤사이 실험·검증, 실험은 한 번에 하나씩, 멀티에이전트, 한 문제·trace·모델·상황에 오버피팅하지 말고 일반화된 단순하고 일관된 개념에서 출발." 그리고 "1번(확실한 결함)은 먼저 고친다."

## 0. "1번 결함"은 고칠 것이 없었다 (2026-11-22 확인)

2026-11-21 대화에서 "보냈지만 엔진에 아직 안 잡힌 요청이 KV·batch 계산에서 빠진다"고 했던 것은 틀렸다. 스케줄러는 요청을 보낼 때 장부(`AddRequestLocalAccount`, `pkg/scheduler/policy/scheduling_policy.go:425`)에 적고, neutral 요청은 prefill 쪽과 decode 쪽 장부 **양쪽에** 더해진다(`pkg/cms/instance_status_local_account.go:220-227`). `decodeBatchOf`(`fluidserve.go:808`)가 decode 쪽 장부를 KV·batch에 더하고, prefill 대기열(`fluidserve.go:1211`)은 prefill 쪽 장부를 더한다. 장부 기능(`--enable-instance-status-local-account`)은 기본값 켜짐이다. 실측: v0.6 8B run(`261120_0230_exp170hr1_fsg10dl_shift62`)에서 인스턴스당 아직 안 잡힌 요청은 표본 평균 0.2건, p90 0건, 최대 52건. 따라서 고칠 결함이 없고 그 실험은 하지 않는다. 같은 이유로 EXP-169 §(swe +4~5 ms)의 "같은 초에 보낸 prompt가 대기 prefill에 없다"는 설명도 틀렸다 — 그 run의 pace 예측은 `max(대기 prefill, h 동안 기대 도착)`에서 기대 도착 쪽이 거의 항상 이기므로(대기가 이긴 경우 0.0 %), 순간적으로 몰리는 도착을 평균으로만 본 것이 더 가까운 설명이다.

## 1. 구성

8B 함대(EXP-170·172와 같다: Llama-3.1-8B × 8, TP=1, `--max-num-seqs 4096`, 게이트웨이 `-v 0`, trace `dyn60_shift_m2Am1B_b1045_x620.csv`, 36 프로세스). **기준은 v0.6(`fsg10dl`, 상한 켬)이고 각 arm은 v0.6에서 하나만 바꾼다.** v0.6은 다시 돌리지 않는다(EXP-170 r1·r2 기록과 세션 간 문턱: 모든 도착·거절률·클래스별 모든 도착 max(4.6, 범위), admitted·클래스별 admitted max(1, 범위), goodput max(5 %, 범위)). 드라이버 `k8s/exp07/run_exp172_hour.sh`(일반 arm `fsx*`, `prep_sched`), 체인 `/home/nxclab/scratch/exp173/run.sh`(대기열), 로그 `/home/nxclab/tools/exp173_chain.log`. 끝나면 70B 설정·Sionna로 되돌린다.

## 2. arm F1: 과거 h 동안 들어온 양을 투영에 더하기 (`fsxah1`, 플래그만, 스케줄러 `7dc2355c`)

사용자의 제안을 그대로 시험한다: 투영 = 지금 KV + 처리 중 요청의 h 동안 성장 − h 동안 끝날 KV **+ 이 인스턴스에 지난 h 동안 route된 prompt 양**(`--fluidserve-arrival-horizons=1`, EXP-124에서 만든 기존 플래그; 새 조정값 없음 — 창은 h 하나). 끝날 양을 크레딧하는 만큼 들어올 양을 같은 구간으로 같이 센다. EXP-124(8B 정적 trace, 1회)에서 거절을 늘리지 않고 admitted 90.0 → 96.9였고 한 시간 trace에서는 처음이다.

**판정(1회 먼저, 결과가 v0.6과 문턱 넘게 다르면 2회차)**: 아홉 지표와 goodput 각각 v0.6 대비. **반증**: preemption이 v0.6(엔진 카운터 53.5k / 53.7k)보다 30 % 넘게 줄지 않으면 도착 항이 KV 수준을 바꾸지 못한 것; 모든 도착이 v0.6보다 4.6점 넘게 낮고 chat 거절이 늘면 수준 판정(EXP-114 §13.42)의 재현. 같이 볼 것: 엔진 KV ≥ 0.973 비율, 인스턴스별 도착 항 크기의 시간 변화(오르내림이 있는지 — 순환의 증거).
