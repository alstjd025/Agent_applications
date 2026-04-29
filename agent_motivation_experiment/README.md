# Motivation Experiment: "The Illusion of Efficiency"

LLM 서버 throughput peak 시점에서 job-level goodput이 붕괴함을 실증.

핵심 메시지:
1. **Throughput peak != useful work** — call-level goodput은 완만히 감소하나 job-level goodput은 급격히 collapse
2. **Abandonment cost** — chain 후반 call일수록 SLO violation rate 높음
3. **Wasted Compute** — throughput peak에서 wasted compute ratio도 peak

---

## 파일 구조

```
agent_motivation_experiment/
├── run_experiment.py            # 메인 실행 스크립트 (all-in-one)
├── synthetic_coding_agent.py   # Stage-based simulated coding agent
├── metrics_tracker.py          # 메트릭 수집 (CSV + TBT JSONL)
├── analyze_motivation.py       # Post-hoc 분석 + 4개 그래프 생성
├── agent_logger.py             # 에이전트 call별 상세 로그
├── prompts/
│   ├── system_prompt.py        # ~14.5K token 코딩 에이전트 시스템 프롬프트
│   ├── stage_prompts.py        # Stage별 instruction 프롬프트
│   └── simulated_artifacts.py  # 도구 결과 시뮬레이션 (파일 내용, 테스트 출력)
└── results/
    ├── baseline_20260424-180204/
    ├── poisson_0.005_20260426-HHMMSS/
    ├── rpm_3_20260426-HHMMSS/
    └── ...
```

---

## 워크로드: Stage-Based Simulated Coding Agent

실제 코딩 에이전트(Claude Code, Devin) 워크플로우 모방:

```
Understand(1) → Locate(1-4) → Plan(1) → Implement(1-3) → Verify(1)
→ [실패 시 Debug → Implement → Verify 루프 반복]
```

| Stage | Call 수 | 도구 결과 (context growth) |
|-------|---------|---------------------------|
| Understand | 1 (고정) | 검색 결과 (~350 tokens) |
| Locate | 1-4 (가변) | 파일 내용 (~640-1550 tokens/call) |
| Plan | 1 (고정) | 없음 |
| Implement | 1-3 (가변) | 없음 |
| Verify | 1 (고정) | 테스트 출력 (~500-785 tokens) |
| Debug | 0-N (잔여 call) | 파일 + 테스트 출력 |

핵심 속성:
- **Chain length**: `randint(5, 30)` — uniform 분포
- **System prompt**: ~14.5K tokens, 모든 request에서 동일 (prefix KV cache 공유)
- **Nonce**: user message에 `[Run ID: {nonce}]`로 주입 → non-prefix KV cache reuse 방지
- **All-or-nothing**: 모든 call이 성공해야 job 완료
- **재현성**: temperature=0.0, seed=42, SGLang `--random-seed 42`

---

## 실험 구성

### Phase 0: Baseline (완료)

| 항목 | 값 |
|------|-----|
| Mode | `--mode single --concurrency 1` |
| Jobs | 164개 (SWE-bench Lite, 300개 중 baseline latency 확보된 것) |
| 결과 | 162 성공, 2 실패 |
| Latency | min=70.5s, p50=484.5s, p90=1144.8s, max=2996.1s, mean=598.3s |
| 결과 경로 | `results/baseline_20260424-180204/` |

### Phase 1: Poisson Sweep

λ값에 따른 Poisson 도착 프로세스로 job 제출, 60분간 실행.

| λ | 평균 도착 간격 | 분당 도착 | 예상 동시 실행 |
|---|-------------|---------|-------------|
| 0.005 | 200초 | ~0.3/min | 1~2 |
| 0.01 | 100초 | ~0.6/min | 2~4 |
| 0.02 | 50초 | ~1.2/min | 4~8 |
| 0.05 | 20초 | ~3/min | 10~20 |
| 0.1 | 10초 | ~6/min | 20~40 |
| 0.2 | 5초 | ~12/min | 40~80+ |

동시 실행 예상은 baseline 평균 JCT ~598s 기준 추정.

### Phase 2: Rate Sweep

고정 간격 제출, 60분간 실행.

| RPM | 제출 간격 | 예상 동시 실행 |
|-----|---------|-------------|
| 3 | 20초 | 3~5 |
| 6 | 10초 | 6~10 |
| 12 | 5초 | 12~20 |
| 30 | 2초 | 30~50 |
| 60 | 1초 | 60~100+ |

### Phase 3: 분석

`analyze_motivation.py`로 4개 motivation figure 생성:
1. Throughput vs Job Goodput (call-level 대비)
2. Call index별 SLO violation rate (abandonment)
3. WCR vs Throughput (wasted compute)
4. Alpha sensitivity (JSA 변화)

---

## Timeout 정책

| Level | 조건 | 기본값 |
|-------|------|--------|
| Call-level TTFT | 첫 토큰 도착 전 | 120초 |
| Call-level idle | 토큰 간 간격 | 60초 |
| Job-level τ | baseline_latency × τ | τ=2.0 |

- τ timeout 초과 시 `is_job_timeout=True`, job 즉시 abort
- 실험 duration 만료 시 `server_terminated_event` set → in-flight job abort
- 서버 connection reset/broken pipe 감지 → `is_server_terminated=True`

---

## SLO 정의

### Call-level SLO
```
Call SLO = (TTFT ≤ TTFT_baseline_p95 × α) AND (TBT_p95 ≤ TBT_baseline_p95 × α)
```
- α 기본값: 1.5 (sensitivity: {1.0, 1.5, 2.0, 3.0})

### Job-level SLO
```
T_SLO(n) = sum(baseline per-call latencies for n calls) × (1 + α_job)
```
- α_job 기본값: 1.0 (sensitivity: {0.5, 1.0, 2.0, 5.0})

### 분석 지표

| 메트릭 | 계산식 | 설명 |
|--------|--------|------|
| Throughput | sum(output_tokens) / wall_time | 서버 출력 처리량 |
| JCR | completed_jobs / submitted_jobs | Job Completion Rate |
| JSA | on_time_jobs / completed_jobs | Job SLO Attainment |
| Job Goodput | JCR × JSA | Job-level useful work 비율 |
| WCR | wasted_tokens / total_tokens | Wasted Compute Ratio |

---

## 실행 방법

### 전체 Poisson Sweep
```bash
/home/nxc/mskim/agent/Agent_applications/agent_sglang_concurrent/.venv/bin/python \
  run_experiment.py \
  --mode poisson-sweep \
  --baseline-dir results/baseline_20260424-180204 \
  --duration-min 60 \
  --tau 2.0
```

### 전체 Rate Sweep
```bash
/home/nxc/mskim/agent/Agent_applications/agent_sglang_concurrent/.venv/bin/python \
  run_experiment.py \
  --mode rate-sweep \
  --baseline-dir results/baseline_20260424-180204 \
  --duration-min 60 \
  --tau 2.0
```

### 개별 Poisson 테스트
```bash
python run_experiment.py \
  --mode single --lambda-val 0.01 \
  --baseline-dir results/baseline_20260424-180204 \
  --duration-min 10 --tau 2.0 --no-server-restart
```

### 개별 Rate 테스트
```bash
python run_experiment.py \
  --mode single --rpm 6 \
  --baseline-dir results/baseline_20260424-180204 \
  --duration-min 10 --tau 2.0 --no-server-restart
```

### 분석
```bash
python analyze_motivation.py \
  --results-dir results \
  --baseline-dir results/baseline_20260424-180204
```

---

## CLI 인자 전체 목록

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--mode` | (필수) | `baseline`, `sweep`, `rate-sweep`, `poisson-sweep`, `single` |
| `--baseline-dir` | None | Baseline metrics.csv 경로 (sweep/rate/poisson 필수) |
| `--duration-min` | 60 | 실험 지속 시간 (분) |
| `--tau` | 2.0 | Job timeout 배수 (baseline_latency × τ) |
| `--lambda-list` | "0.005,0.01,0.02,0.05,0.1,0.2" | Poisson sweep λ값 목록 |
| `--lambda-val` | None | Single Poisson λ값 |
| `--rate-list` | "3,6,12,30,60" | Rate sweep RPM 목록 |
| `--rpm` | None | Single rate RPM값 |
| `--concurrency` | 8 | 동시 실행 job 수 (baseline/sweep) |
| `--dataset` | "princeton-nlp/SWE-bench_Lite" | HuggingFace dataset |
| `--server-base-url` | "http://localhost:8080" | SGLang 서버 URL |
| `--seed` | 42 | 재현성 seed |
| `--chain-min` | 5 | 최소 chain length |
| `--chain-max` | 30 | 최대 chain length |
| `--output-dir` | "results" | 결과 출력 디렉토리 |
| `--sglang-ssh-host` | None | SGLang 서버 SSH host (예: NXC7) |
| `--sglang-start-cmd` | (기본값 있음) | 서버 시작 명령어 |
| `--sglang-stop-cmd` | (기본값 있음) | 서버 중지 명령어 |
| `--sglang-tmux-session` | "sglang" | 원격 tmux 세션명 |
| `--no-server-restart` | False | 실험 간 서버 리스타트 스킵 |
| `--resume-dir` | None | Baseline 재개용 디렉토리 |
| `--log-level` | "quiet" | `quiet`, `info`, `debug` |

---

## 결과 디렉토리 구조

각 실험 실행 시 태그+타임스탬프로 디렉토리 자동 생성:

```
results/
├── baseline_20260424-180204/          # Baseline
├── poisson_0.005_20260426-040840/     # Poisson λ=0.005
├── poisson_0.01_20260426-XXXXXX/      # Poisson λ=0.01
├── ...
├── rpm_3_20260426-XXXXXX/             # Rate 3 RPM
├── rpm_6_20260426-XXXXXX/             # Rate 6 RPM
├── ...
└── single_20260426-XXXXXX/            # 개별 테스트
```

### 각 실험 디렉토리 내용

```
poisson_0.01_20260426-XXXXXX/
├── run_config.json        # 실험 설정 (mode, lambda, tau, duration 등)
├── metrics.csv            # 핵심 메트릭 (chain_call + job_summary 혼합)
├── errors.log             # 에러 로그
├── tbt_events.jsonl       # TBT 상세 이벤트 (per-chunk timestamps)
└── agent_logs/            # 에이전트 call별 상세 로그
    ├── <task_id>_call_001.json
    ├── <task_id>_call_002.json
    └── ...
```

### run_config.json 예시

```json
{
  "mode": "poisson-sweep",
  "lambda": 0.01,
  "duration_min": 60.0,
  "tau": 2.0,
  "chain_range": [5, 30],
  "dataset": "princeton-nlp/SWE-bench_Lite",
  "server_base_url": "http://localhost:8080",
  "seed": 42,
  "baseline_dir": "results/baseline_20260424-180204",
  "created_at": "2026-04-26T04:08:40.123456"
}
```

---

## metrics.csv 스키마

모든 실험 결과가 동일한 CSV 스키마를 공유. `agent` 컬럼으로 row 타입 구분.

### Row 타입 구분

| agent 값 | 의미 |
|----------|------|
| `chain_call_understand` | Understand stage LLM call |
| `chain_call_locate` | Locate stage LLM call |
| `chain_call_plan` | Plan stage LLM call |
| `chain_call_implement` | Implement stage LLM call |
| `chain_call_verify` | Verify stage LLM call |
| `chain_call_debug` | Debug stage LLM call |
| `job_summary` | Job 단위 요약 |

### 전체 컬럼

| 컬럼 | 타입 | chain_call | job_summary | 설명 |
|------|------|:----------:|:-----------:|------|
| `task_id` | string | O | O | Job ID (예: `astropy__astropy-14365__replay01`) |
| `iteration` | int | O | O | 반복 번호 |
| `agent` | string | O | O | Row 타입 (위 테이블 참조) |
| `call_index` | int | 1-based | calls_completed | call 번호 / 완료 call 수 |
| `total_calls_expected` | int | O | chain_length | 예상 총 call 수 |
| `start_time` | float | O | O | 시작 epoch (call/job) |
| `end_time` | float | O | O | 종료 epoch (call/job) |
| `latency` | float | O | O | 소요 시간 (초) |
| `input_tokens` | int | O | O | 입력 토큰 수 |
| `output_tokens` | int | O | O | 출력 토큰 수 |
| `first_token_latency` | float | O | null | TTFT (초) |
| `decode_speed_tps` | float | O | null | 디코딩 속도 |
| `gpu_memory_mb` | float | O | null | GPU 메모리 사용량 |
| `kv_cache_usage_pct` | float | O | null | KV cache 사용률 |
| `transition_time` | float | O | null | Stage 전환 시간 |
| `tokenizer_mode` | string | O | null | 토크나이저 모드 |
| `stream_fallback_used` | bool | O | false | 스트리밍 폴백 여부 |
| `tbt_available` | bool | O | null | TBT 데이터 가용성 |
| `stream_chunks` | int | O | null | 스트리밍 청크 수 |
| `streamed_output_tokens_est` | int | O | null | 스트리밍 출력 토큰 추정 |
| `first_chunk_tokens_est` | int | O | null | 첫 청크 토큰 수 |
| `tbt_mean_ms` | float | O | null | 평균 TBT |
| `tbt_p50_ms` | float | O | null | p50 TBT |
| `tbt_p75_ms`~`tbt_p95_ms` | float | O | null | TBT 백분위수 |
| `tbt_max_ms` | float | O | null | 최대 TBT |
| `tbt_sample_count` | int | O | null | TBT 샘플 수 |
| `is_timeout` | bool | O | null | Call-level timeout |
| `is_error` | bool | O | null/bool | Call-level error |
| `is_job_timeout` | bool | O | O | Job τ timeout |
| `job_timeout_sec` | float | O | O | τ timeout 한계 (초) |
| `is_server_terminated` | bool | O | O | 서버 종료로 abort |
| `job_submit_time` | float | O | O | Job 제출 epoch |
| `job_end_time` | float | O | O | Job 종료 epoch |
| `job_completed` | bool | null | O | Job 전체 완료 여부 |
| `concurrency_level` | int | O | O | 관측 동시성 |
| `success` | bool | O | null | Call 성공 여부 |
| `error_msg` | string | O | O | 에러 메시지 |
| `timestamp` | string | O | O | 기록 시각 (ISO) |

---

## 서버 환경

| 항목 | 설정 |
|------|------|
| Model | meta-llama/Llama-3.3-70B-Instruct |
| GPU | 4×B200 (NXC7) |
| Access | SSH tunnel localhost:8080 → NXC7:31000 |
| SGLang flags | `--random-seed 42` |
| Max context | 131,072 tokens |
| System prefix | ~14.5K tokens |
| Server start | `ssh NXC7` → `cd /home/nxclab/sglang/ms_dev/expctl && python3 run_pd_experiment.py --mode single --single-port 31000` |
| Server stop | `ssh NXC7` → `bash /home/nxclab/sglang/ms_dev/stop_servers.sh` |

### 서버 자동 제어 (--sglang-ssh-host)

`--sglang-ssh-host NXC7` 지정 시:
- 실험 간 SSH로 서버 stop/start 자동 실행
- KV cache 초기화 보장
- `--no-server-restart` 시 스킵 (서버 이미 실행 중인 경우)

---

## 진행 상태 확인

```bash
# tmux 세션 접속
tmux a -t motivation

# 완료된 job 수
for d in results/*/; do echo "$d: $(grep -c job_summary ${d}metrics.csv 2>/dev/null || echo 0) jobs"; done

# 특정 실험 설정 확인
cat results/poisson_0.01_*/run_config.json
```
