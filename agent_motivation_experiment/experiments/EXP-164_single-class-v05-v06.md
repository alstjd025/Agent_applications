# EXP-164: 클래스 하나만 있는 한 시간 trace — v0.5 대 v0.6

**날짜**: 2026-11-18 (실행 전 작성)
**질문**: 클래스가 하나뿐일 때(chat만 / swe만 / deepresearch만) v0.6이 v0.5보다 나은가. EXP-163(v0.6만, 보정)에서 v0.6은 swe만에서 판정선 75 ms로 받았는데 실제 iteration이 79 ms였고(admitted 97.91),
deepresearch만에서 KV가 묶여 preemption 11,576회와 첫 토큰 지연이 났다(admitted 97.19). 이 두 약점이 v0.5에도 있는지, v0.6에서 생긴 것인지를 같은 세션에서 가린다.

## 구성 (2026-11-18 사용자 결정: chat은 도착률을 올리고 swe·deepresearch는 그대로, v0.5와 v0.6 모두)

| 변형 | trace | 워크로드 설정 | 부하 생성 프로세스 |
|---|---|---|---|
| `onlychat25` | `dyn60_onlychat_b25112.csv` — 같은 생성기·시드·부하 모양으로 도착률만 2.5배(초당 25~112.5건, 분 단위 25.7~111.7), 모두 chat, 249,234건. 생성 명령은 10/45로 주면 원래 trace를 바이트 단위로 재현한다(확인함) | `mix_dyn60_onlychat_b25112_t75.json` | **36** (EXP-114: 12개로는 6.2배 trace를 다 내지 못했다; 2.5배 정점에 동시 chat 스트림 약 2,000) |
| `onlyswe` | `dyn60_onlyswe_b1045.csv` (EXP-163과 같다) | `mix_dyn60_onlyswe_b1045_t75r.json` (반복 번호 켬) | 12 |
| `onlydr` | `dyn60_onlydr_b1045.csv` (EXP-163과 같다) | `mix_dyn60_onlydr_b1045_t75.json` | 12 |

chat 기록 34,513개는 2.5배에서 평균 약 8분(정점 약 5분)마다 다시 와 엔진 prompt 유지 시간·스케줄러 prefix index보다 길다 — 반복 번호 없음.

| 항목 | 값 |
|---|---|
| arm | `fsexactprog`(v0.5), `fsg10dl`(v0.6) |
| 바이너리 | 스케줄러 `7dc2355c`, 게이트웨이 `05745270`, h100 |
| 순서 | 반복이 바깥: r1 = onlychat25(v0.5 → v0.6), onlyswe(v0.5 → v0.6), onlydr(v0.5 → v0.6); r2 = 같은 변형 순서, arm 순서 반대. 12 run, 약 14.5시간 |
| 드라이버·체인 | `k8s/exp07/run_exp164_hour.sh`(변형별 설정 검사, onlychat25만 프로세스 36), `/home/nxclab/scratch/exp164/run.sh`(run마다 `run_config.json`의 클래스 계획·swe ttft 7000·반복 번호·`load_procs` 확인, 아니면 WARN), 로그 `/home/nxclab/tools/exp164_chain.log` |

## 판정 (실행 전에 정함; 변형마다 따로)

1. 지표: 모든 도착, admitted, 거절률(클래스가 하나라 클래스별 지표는 이것과 같다). "평균 차이 > max(문턱, 두 조건 각각의 반복 범위)"이면 "다르다". 문턱은 모든 도착·거절률 4점, admitted 1점.
2. 같이 적는 것: 받은 요청의 마감 위반 종류(첫 토큰 / 평균 토큰 간격 / 몰림), TTFT p50/p90/p99, 평균 토큰 간격 분포와 budget 초과 비율, iteration 실측(인스턴스 판정선 대비), preemption, goodput,
   첫 토큰 예측 대 실제, prefix 적중, 결정 지연, 클라이언트 상태(분별 요청 수 대 trace — onlychat25에서 특히).
3. 해석의 초점: (가) swe만에서 v0.5의 iteration이 판정선(0.9 × 75 = 67.5 ms) 아래에 머무는가, 그리고 admitted가 v0.6보다 높은가 — "v0.5의 0.9가 판정 시점 추정과 실제의 차이를 흡수한다"는 추론의 검증.
   (나) deepresearch만에서 v0.5도 preemption·첫 토큰 지연을 보이는가 — KV 판정(끝점 투영, 물리 KV × 0.95)은 두 구성이 같으므로 비슷하리라 예상.
   (다) chat 2.5배에서 두 구성이 admission이 움직이는 부하에 들어가는가.

**예상**: swe만에서는 v0.5의 admitted가 v0.6보다 높고 거절률도 높다. deepresearch만에서는 두 구성이 비슷하게 KV에 묶인다. chat 2.5배에서는 v0.6이 더 받고 admitted는 비슷하다.

## 결과

(실행 후 적는다)
