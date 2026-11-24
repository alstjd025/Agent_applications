# EXP-178 — 후보 D3(KV 가격 + GPU 시간 가격)를 8B 표준 budget과 70B에서 (2026-11-24)

**사용자 지시(2026-11-24 10:50)**: "D3를 표준에서 해보고, 70B에서도 바로 해봐."

**물음**: EXP-177에서 표준 budget(8B)에서는 KV 가격만 쓰는 A가 좋았고(88.41 / 88.64), 절반 budget에서는 GPU 시간 가격(T 87.28)과 두 가격을 같이 쓰는 D3(87.43)가 좋았다.
GPU 시간 가격만 쓰는 T는 표준 budget에서 A보다 11.9점 낮았다. 두 가격을 같이 두는 D3가 **하나의 정책으로 두 조건(그리고 70B)을 모두 덮는가**.

**D3 (arm `fsxdual`, 바이너리 `scheduler-exp177-dual` `1de94045`, 브랜치 `exp177-dual` `3ba37e7`)** = A(물리 KV stock 메모리 판정 + 받지 못한 수요로 오르는 KV 가격 λ)
+ T의 GPU 시간 가격 μ. 판정 λ × c + μ × a ≤ 1. 정의는 `EXP-177_overnight-candidates.md`의 A·T·D3 절.

## 구성

| | 8B 표준 budget (EXP-178) | 70B (EXP-178b) |
|---|---|---|
| 함대 | 8 × Llama-3.1-8B TP1 | 4 × Llama-3.1-70B TP2 |
| trace | `shift62`(약 61만 도착), 부하 생성 36 | `shift`(약 9.8만 도착), 부하 생성 12 |
| budget | chat 5 s/50 ms, dr 10 s/100 ms, swe 7 s/75 ms | 같음 |
| 드라이버 | `run_exp172_hour.sh` | `run_exp175_hour.sh` |
| 체인 | `/home/nxclab/scratch/exp178/run.sh`, 로그 `/home/nxclab/tools/exp178_chain.log` | `/home/nxclab/scratch/exp178b/run.sh`(8B 체인 끝을 기다림), 로그 `exp178b_chain.log` |
| gateway | `3590c1af` | `3590c1af` |
| 반복 | 2 | 2 |
| 채점 | ladder95, 대시보드 점수 | 같음 |

## 판정 규칙(실행 전)

비교 기준은 다른 세션의 기록이므로 세션 간 문턱(모든 도착·거절률·클래스별 모든 도착 max(4.6, 범위), admitted max(1, 범위), goodput max(5 %, 범위))으로 지표마다 판정한다.

**8B 표준 budget** — 기준: A(EXP-177, 2회: 88.41 / 88.64, admitted 99.68 / 99.75), PolyServe(EXP-174, 2회 77.95), 후보 3(EXP-174, 2회 83.94).
- 예측: A와 어느 지표도 다르지 않다(이 조건에서 받는 결정을 막는 것은 KV이고, μ는 GPU 시간이 묶일 때만 커진다). admitted ≥ 99, preemption 1만 미만.
- 반증: 모든 도착이 A보다 4.6점 넘게 낮다(< 83.9) → 두 가격을 같이 두면 KV가 한계인 조건에서 GPU 시간 가격이 해를 끼친다(같은 요청을 두 가격이 중복으로 벌하는 위험, 에이전트 ③이 적은 실패).

**70B** — 기준: v0.6(EXP-161, 2회 80.69, admitted 99.90), 후보 3(EXP-175, 2회 82.62), cap만 끈 것(EXP-175, 2회 81.08), PolyServe(EXP-175, 2회 53.80).
- 예측: v0.6·후보 3과 어느 지표도 다르지 않거나 더 낫다(70B는 KV가 묶이지 않아 λ ≈ 0; GPU 시간이 묶이지만 v0.6의 gate가 이미 한계에서 운영한다). admitted ≥ 99.5.
- 반증: 모든 도착이 v0.6보다 4.6점 넘게 낮다(< 76.1), 또는 admitted가 1점 넘게 낮다 → D3는 70B에서 나빠진다.
- 의도적으로 포기하는 것: deepresearch(요청당 자원을 가장 많이 씀)의 거절 증가. 실패와 구분해 적는다.
