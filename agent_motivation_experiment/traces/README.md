# traces/ — arrival-trace preprocessing

Preprocessing area for **arrival timing** used by
`run_experiment.py --mode trace-replay`. Public traces are downloaded here,
converted to the canonical format, and (optionally) scaled up/down. The
runner only ever reads the canonical format — it never knows the source trace.

- Format contract: [TRACE_FORMAT.md](TRACE_FORMAT.md)
- Runner-side reader: `../arrival_trace.py`

## Layout

```
traces/
├── TRACE_FORMAT.md       # canonical format spec (the contract)
├── scale_trace.py        # up/down scale any canonical trace (source-agnostic)
└── azure/                # one folder per source
    ├── convert_azure.py      # Azure LLM Inference -> canonical
    ├── analyze_azure.py      # count / span / token & rate / burstiness stats
    ├── plot_azure_rate.py    # hourly request-count curve (PNG)
    ├── raw/                  # downloaded public traces  (gitignored: *.csv)
    ├── canonical/            # converted + scaled traces (gitignored: *.csv)
    └── plots/                # figures (+ hourly-count caches, *.csv gitignored)
```

`*.csv` is gitignored repo-wide, so trace **data** is not version-controlled —
only the scripts and specs are. Regenerate data with the recipes below (or
`git add -f` a specific canonical file to pin it to a run).

## Azure LLM Inference — download, convert, analyze

Two datasets are on GitHub (schema `TIMESTAMP,ContextTokens,GeneratedTokens`):

- **2023** (Splitwise): `data/AzureLLMInferenceTrace_{code,conv}.csv` in-repo.
  Code = 8.8K reqs / ~57 min, bursty (peak/mean ≈ 3.3×).
- **2024** (DynamoLLM): release `dataset-llm-2024`, `*_1week.csv` (code 692 MB /
  16.8M reqs, conv 1.13 GB / 27.3M reqs). Each spans **7 days** (code May 10–16,
  conv May 12–18 UTC). code is prefill-heavy + very diurnal (hourly CV 0.85);
  conv is decode-heavy + steadier (CV 0.29).

```bash
# 2024 (large; from GitHub release)
curl -sSL -o traces/azure/raw/2024/AzureLLMInferenceTrace_conv_2024.csv \
  https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv

# inspect: what data / how long / token & rate / burstiness
python3 traces/azure/analyze_azure.py traces/azure/raw/2024/AzureLLMInferenceTrace_conv_2024.csv

# hourly request-count curve
python3 traces/azure/plot_azure_rate.py \
  conv2024=traces/azure/raw/2024/AzureLLMInferenceTrace_conv_2024.csv \
  --out traces/azure/plots/azure_2024_hourly_rate.png --unit day

# convert a window to canonical (raw timestamps). --window-min slices minutes.
python3 traces/azure/convert_azure.py \
  traces/azure/raw/2024/AzureLLMInferenceTrace_conv_2024.csv \
  traces/azure/canonical/conv2024_peak1h.csv --window-min 3720 3780
```

`convert_azure.py` also has upstream intensity knobs (`--time-scale`,
`--target-lambda`, `--window-min`); prefer `scale_trace.py` (below) for
load scaling that preserves the burst pattern.

## Scaling a canonical trace up or down (`scale_trace.py`)

`--factor f` multiplies offered load, preserving the temporal/burst pattern.
Source-agnostic — operates on any canonical trace.

- **f ≥ 1 UPSCALE** — TraceUpscaler overlay (EuroSys '24 port): SAME request
  count, uses first ~1/f of the source. ⇒ rate ×f, **duration ÷f**.
- **f < 1 DOWNSCALE** — TraceUpscaler is upscale-only; use a standard inverse:
  - `--down-method thin` (default): Bernoulli thinning, keep each request
    w.p. f. ⇒ rate ×f, **duration unchanged**, count ×f. Best for matching a
    big-cluster trace's real window to a smaller server's capacity.
  - `--down-method stretch`: dilate time by 1/f, keep every request. ⇒ rate
    ×f, duration ÷f (longer), count preserved.

```bash
# downscale sweep (constant 60-min window; good for steady-state goodput)
for F in 0.125 0.25 0.5; do
  python3 traces/scale_trace.py traces/azure/canonical/conv2024_peak1h.csv \
    traces/azure/canonical/conv2024_peak1h_thin$F.csv --factor $F
done
# upscale beyond base (duration shrinks with f)
python3 traces/scale_trace.py traces/azure/canonical/conv2024_peak1h.csv \
  traces/azure/canonical/conv2024_peak1h_x4.csv --factor 4
```

## Driving a run

```bash
python run_experiment.py \
  --workload codingagent_request_level_poisson \
  --mode trace-replay \
  --node nxc7-1 \
  --transcript-file results/transcripts/swe_b200x2.jsonl \
  --trace-file traces/azure/canonical/conv2024_peak1h_thin0.25.csv \
  --tau 3.0 \
  --session-name trace_conv_thin0p25
```

## Adding another source (BurstGPT, Mooncake, …)

Make `traces/<source>/convert_<source>.py` that reads the source schema and
emits the canonical columns (`arrival_s` required). See TRACE_FORMAT.md
§"Producing a canonical file". `scale_trace.py` and the runner need no changes.
