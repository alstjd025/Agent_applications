"""Canonical arrival-trace reader for `--mode trace-replay`.

The runner drives request/job arrival *timing* from a trace file. Trace
preprocessing (per-source conversion, scaling, windowing) lives in `traces/`
and emits the canonical format documented in `traces/TRACE_FORMAT.md`:

    CSV with a header; one row per arrival; required column `arrival_s`
    (float seconds, relative to trace start). Optional columns
    (`request_id`, `input_tokens`, `output_tokens`, ...) are ignored here.

This module only *reads* that format — it never rescales timing. Returns a
sorted list of arrival offsets in seconds, zeroed so the run starts
immediately. `cap_min` optionally drops arrivals past `cap_min` minutes.
"""

import csv
from typing import List, Optional


def load_arrival_trace(path: str, cap_min: Optional[float] = None) -> List[float]:
    """Load arrival offsets (seconds from start) from a canonical trace CSV.

    - Requires an `arrival_s` column; raises ValueError otherwise.
    - Sorts ascending and subtracts the minimum so the first arrival is at 0
      (no leading idle), without otherwise rescaling.
    - `cap_min`: if given, drops arrivals with offset > cap_min*60 seconds.
    """
    offsets: List[float] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "arrival_s" not in reader.fieldnames:
            raise ValueError(
                f"Trace file {path!r} is missing required 'arrival_s' column "
                f"(found columns: {reader.fieldnames}). "
                f"See traces/TRACE_FORMAT.md."
            )
        for lineno, row in enumerate(reader, start=2):
            raw = (row.get("arrival_s") or "").strip()
            if raw == "":
                continue
            try:
                offsets.append(float(raw))
            except ValueError:
                raise ValueError(
                    f"Trace file {path!r} line {lineno}: "
                    f"arrival_s={raw!r} is not a float."
                )

    if not offsets:
        raise ValueError(f"Trace file {path!r} contains no arrivals.")

    offsets.sort()
    base = offsets[0]
    offsets = [t - base for t in offsets]

    if cap_min is not None:
        cap_s = cap_min * 60.0
        offsets = [t for t in offsets if t <= cap_s]
        if not offsets:
            raise ValueError(
                f"Trace file {path!r}: no arrivals within --trace-duration-min "
                f"({cap_min} min)."
            )

    return offsets
