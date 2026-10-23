# Figure 1 scripts (FluidServe)

Needs Python 3 with numpy, pandas, matplotlib. Run each script from anywhere;
outputs are written next to the script (PDF at final column width, plus PNG).

```
python3 scatter/fig_goodput_vs_attainment.py             # Fig. 1 scatter
python3 scatter/fig_goodput_vs_attainment.py --chat-rr   # labels with Chat RR
python3 scatter/fig_goodput_vs_attainment_byclass.py     # (a) all / (b) Chat only
python3 operating_space/fig_operating_space_data.py      # measured envelope
python3 operating_space/fig_operating_space.py           # model figures (2D, 3D)
```

| Script | Figure | Data |
|---|---|---|
| `scatter/fig_goodput_vs_attainment.py` | token goodput (x) vs SLO attainment of admitted requests (y), one point per scheduler | `data/verdicts/` (measured) |
| `scatter/fig_goodput_vs_attainment_byclass.py` | the same, all requests vs. Chat only | `data/verdicts/` (measured) |
| `operating_space/fig_operating_space_data.py` | per-request TBT vs. concurrent decoding requests; class budgets cross the envelope at the admissible batch | `data/trace/` (measured) |
| `operating_space/fig_operating_space.py` | schematic 2D / 3D operating space | **model, not measured** (`lat()` in the script is a stand-in; replace it with a profile) |

## Data (EXP-109, Llama-3.1-70B, 4 instances, one-hour shifting trace)

- `data/verdicts/*.csv`: per-request ladder95 verdicts, 5 schedulers x 2
  repeats. A request meets its SLO if >= 95% of its tokens are on time (token
  i is on time within TTFT budget + i x TBT budget; Chat 5 s / 50 ms, Agent
  7 s / 75 ms, Deep Research 10 s / 100 ms). Columns: class, ladder_ok,
  n_tokens, n_late, rejected, errored, cutoff, rel.
- `data/trace/*.csv`: raw request traces (first 12-18 min of each run) of
  vLLM r1/r2 and Llumnix r2, used for the TBT envelope.

Run-name key: `vllmcache` vLLM router, `slot` Llumnix (SLO-aware policy),
`polyservep` PolyServe, `llmdslot` llm-d, `fsv3capgnofrc` FluidServe.

## Definitions used by the scatter

- x = on-time output tokens per second over 0-59.25 min (admitted, non-errored requests).
- y = met / (arrivals - rejected); a request still running at the end of the run counts as admitted and not met (only vLLM has any).
- RR = rejection ratio = rejected / arrivals. Marker = repeat 2.

## Notes

- Type is Nimbus Roman No9 L, bundled in `fonts/`; if it is missing the
  scripts fall back to Times New Roman.
- Colours of the five systems are the paper's ramp; llm-d's lavender `#b3a2c9`
  was matched by eye to Fig. 10 and should be replaced by that script's value.
- `paper_style.py` is the shared style file (one copy per folder, identical).
