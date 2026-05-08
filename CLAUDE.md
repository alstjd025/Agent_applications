# Agent Notes (repo root)

이 저장소(`alstjd025/Agent_applications`)는 agent workload 부하 실험 코드를 담습니다.
핵심은 [`agent_motivation_experiment/`](agent_motivation_experiment/) — SGLang server throughput이
멀쩡해 보여도 **application/job-level goodput이 무너지는 구간**을 SWE-bench Lite 기반
synthetic coding agent workload로 측정합니다.

이 파일은 인덱스이고, 실제 실험 규칙·명령은 하위 문서가 더 자세합니다.

---

## Repo Layout

```
.
├── CLAUDE.md                          # 이 파일
├── .gitignore                         # .venv/, __pycache__/, *.pyc, *.log, *.csv
├── agent_motivation_experiment/       # ★ 메인 프로젝트. 자체 README/CLAUDE.md/AGENTS.md 보유
├── L40s_forwarding.md                 # L40S 노드용 SSH 포트포워딩 메모
└── safe_push.sh                       # 공유 GitHub 계정에서 push 시 cached creds 먼저 reject 후 push
```

> 참고: 이 repo는 부모 디렉토리 `/home/nxc/mskim/agent/` 안에 위치합니다.
> 부모 쪽에는 결과 백업(`backup/`), 샘플 metrics(`swe_bench_test_data/`), 수동 SSH 터널 스크립트
> (`pipe_sglang.sh`, `kill_pipe_sglang.sh`) 등이 있지만 **버전관리 대상이 아닙니다.**

---

## Where to Look First

작업 종류별 진입 순서:

- **실험 실행/수정/디버깅** →
  1. [`agent_motivation_experiment/README.md`](agent_motivation_experiment/README.md) (Quick Start, 모드별 명령)
  2. [`agent_motivation_experiment/CLAUDE.md`](agent_motivation_experiment/CLAUDE.md) (core flow, server ownership, admission control 검사 절차)
  3. [`agent_motivation_experiment/AGENTS.md`](agent_motivation_experiment/AGENTS.md)
- **워크로드 어댑터 변경** →
  1. [`agent_motivation_experiment/workloads/AGENTS.md`](agent_motivation_experiment/workloads/AGENTS.md)
  2. 해당 workload 폴더의 `AGENTS.md` (예: `swe_bench_coding_tool_delay/`, `swe_bench_coding_parallel_tool_delay/`)
- **분석 스크립트** → [`agent_motivation_experiment/analysis_scripts/`](agent_motivation_experiment/analysis_scripts/)

---

## Top-level Rules

이 repo 차원에서 통용되는 규칙. 하위 프로젝트가 더 구체적인 규칙을 두면 그쪽이 우선합니다.

- **SGLang 라이프사이클은 `agent_motivation_experiment/run_experiment.py`가 owner.**
  부모 디렉토리의 수동 SSH 터널 스크립트(`pipe_sglang.sh` / `kill_pipe_sglang.sh`)는
  사용자가 명시적으로 저수준 디버깅을 요청한 경우에만 사용. 평소 admission/throughput 의문은
  `run_experiment.py` 의 `single` / `rate-sweep` / `poisson-sweep` 모드로 풀고, 결과 디렉토리
  (`metrics.csv`, `errors.log`, `agent_logs/`, `server_session/`)에서 답을 찾습니다.
- **SGLang 서버는 원격 SSH host `NXC7` 위에서 동작.** 러너가 tmux를 통해 시작/중지/세션 fetch까지 담당.
- **장기 실험은 로컬 tmux `motivation` 세션에서.** 세션이 없으면 `tmux new -s motivation`.
- **Push는 [`safe_push.sh`](safe_push.sh)** — 공유 GitHub 계정 자격증명이 캐시되어 충돌하는 환경이라,
  스크립트가 cached creds를 먼저 reject한 뒤 push.
- **L40S GPU 노드 작업이 필요하면** [`L40s_forwarding.md`](L40s_forwarding.md) 의 SSH 포트포워딩 명령부터 확인.

---

## Operating Conventions (전역)

- 사용자가 명확히 실행/수정/테스트를 요청하면 바로 진행한다.
- 요구가 애매하거나 실험 설정이 결과를 크게 바꿀 수 있으면 먼저 짧게 확인한다.
- 큰 코드 변경은 pseudocode/스켈레톤을 먼저 보여주고 피드백을 받은 뒤 진행한다.
- `metrics.csv`, `tbt_events.jsonl`, `application_*.csv` 등 **run output schema는 임의 변경 금지**.
  새 분석은 새 CSV을 추가하는 방향으로.
- 분석 스크립트를 옮기거나 편집한 뒤에는
  `python -m py_compile analysis_scripts/*.py run_experiment.py` 로 syntax sanity check.

---

## Quick Glossary

- **call goodput** — `call_latency < baseline_call_latency * tau` 인 LLM call 비율
- **job goodput** — `job_latency < baseline_job_latency * tau` 인 multi-call job 비율
- **tau (`--tau`)** — goodput 임계 배수 (보통 3.0)
- **baseline** — concurrency 1로 task별 기준 latency 수집한 run. `--baseline-dir` 로 다른 run에 주입
- **session-name (`--session-name`)** — 로컬 결과 디렉토리 suffix이자, 원격 SGLang 서버의 runtime session 이름. 로컬/원격 로그 매칭 키
- **execution_round** — `swe_bench_coding_parallel_tool_delay` 에서 같은 dependency barrier 뒤에 동시에 출발하는 call 그룹 (`wave` 아님)
- **transition_time** — `tool_delay` workload의 job summary 필드. 그 job에서 application-side로 simulate한 tool call interval 총합
- **NXC7** — 원격 SGLang 서버 SSH host alias

---

## Memory / Auto-context Notes

- 자동 메모리에 저장된 Motivation Experiment Timeout Settings:
  `PER_CALL_TIMEOUT = 120s` (TTFT), `IDLE_TIMEOUT = 60s` — agent_motivation_experiment에서 기준 타임아웃.
- 새 하위 프로젝트가 생기면 이 파일의 **Repo Layout**과 **Where to Look First**를 갱신하세요.
