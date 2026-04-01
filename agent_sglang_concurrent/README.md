# agent_sglang_concurrent

Concurrent LangGraph SWE-bench-mini workload targeting an OpenAI-compatible SGlang server.

Current assumptions:
- SGlang chat endpoint is reachable at `http://localhost:30000/v1`
- Model name is `meta-llama/Llama-3.1-8B-Instruct`
- Server-side Prometheus metrics are optional; request-level CSV logging works without them

Workflow:
- `planning -> coding -> debugging`
- `debugging` must answer in one of:
  - `PASS: ...`
  - `FAIL: ...`
- `FAIL:` causes the workflow to loop back to `planning`

Environment setup:

```bash
./setup_venv.sh
source .venv/bin/activate
```

Run example:

```bash
python run_swebench.py \
  --server-base-url http://localhost:30000 \
  --request-rate-per-min 60 \
  --start-index 0 \
  --end-index 10
```
