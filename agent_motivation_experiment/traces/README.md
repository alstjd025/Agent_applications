# traces/ — arrival-trace preprocessing

Preprocessing area for **arrival timing** used by
`run_experiment.py --mode trace-replay`. Public traces are downloaded here,
converted to the canonical format, and (optionally) intensity-shaped. The
runner only ever reads the canonical format — it never knows the source trace.

- Format contract: [TRACE_FORMAT.md](TRACE_FORMAT.md)
- Runner-side reader: `../arrival_trace.py`

## Layout

```
traces/
├── TRACE_FORMAT.md       # canonical format spec (the contract)
├── convert_azure.py      # Azure LLM Inference  -> canonical
├── raw/                  # downloaded public traces  (gitignored: *.csv)
└── canonical/            # converted canonical files (gitignored: *.csv)
```

`*.csv` is gitignored repo-wide, so trace data is **not** version-controlled —
only the converters and this spec are. Regenerate data with the recipes below
(or `git add -f` a specific canonical file if you want to pin it to a run).

## Azure LLM Inference 2023 (Code workload) — first verification trace

8,819 requests over ~57 min; bursty (peak/mean ≈ 3.3×, per-min CV ≈ 0.79,
~17% simultaneous arrivals). Mean rate ≈ 2.57 req/s.

```bash
# 1) download raw
curl -sSL -o traces/raw/AzureLLMInferenceTrace_code_2023.csv \
  https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv

# 2) convert to canonical (raw timestamps, no scaling)
python3 traces/convert_azure.py \
  traces/raw/AzureLLMInferenceTrace_code_2023.csv \
  traces/canonical/azure_code_2023.csv

# 2b) optional intensity shaping (all upstream of the runner):
#   --time-scale 4         # 4x sparser (denser if <1)
#   --target-lambda 0.5    # normalize mean rate to 0.5 req/s
#   --window-min 0 10      # keep first 10 min only
```

Then drive a run with it (timing from trace, content from the workload):

```bash
python run_experiment.py \
  --workload codingagent_request_level_poisson \
  --mode trace-replay \
  --node nxc7-1 \
  --transcript-file results/transcripts/swe_b200x2.jsonl \
  --trace-file traces/canonical/azure_code_2023.csv \
  --tau 3.0 \
  --session-name trace_azure_code
```

## Adding another source (BurstGPT, Mooncake, …)

Write `convert_<source>.py` that reads the source schema and emits the
canonical columns (`arrival_s` required). See TRACE_FORMAT.md §"Producing a
canonical file" for per-source timestamp handling. The runner needs no changes.
