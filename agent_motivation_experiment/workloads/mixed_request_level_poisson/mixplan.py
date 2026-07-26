"""Deterministic class-mixing plan for the mixed request-level workload.

Two ways to decide which class each arrival belongs to:

**Static mix (`build_class_sequence`).** The mix is integer weights per class,
e.g. {"chat": 1, "deepresearch": 1, "swe": 1}. Arrivals are assigned a class by
repeating a **block** whose composition matches the weights exactly (block size
= Σweights), with the block shuffled deterministically per block index.

Why blocks instead of independent per-arrival sampling: over a 5-minute run
i.i.d. sampling leaves the realised ratio off target by a noticeable margin at
low rates (and the whole point of the experiment is to compare *ratios*).
Block-repeat pins the realised composition to the target within one block,
while the per-block shuffle keeps the classes interleaved rather than arriving
in fixed rotation (which would alias with the arrival process).

**Time-varying mix (`load_class_plan`).** For dynamic-trace runs the mix has to
change during the run, so the plan is precomputed offline — one class per
arrival — and read from the SAME csv the runner replays for arrival timing
(`traces/dynamic/build_dynamic_mix_trace.py`). Row i of the file is arrival i,
so class and time are paired by index and neither side has to agree on a clock.
That generator still builds each segment with `build_class_sequence`, so the
realised ratio is exact per segment exactly as in the static case.
"""

import csv
import random
from typing import Dict, List


def build_class_sequence(weights: Dict[str, int], length: int, seed: int) -> List[str]:
    """Return `length` class labels realising `weights` in shuffled blocks."""
    classes: List[str] = []
    for name, w in sorted(weights.items()):
        w = int(w)
        if w < 0:
            raise ValueError(f"negative weight for {name!r}")
        classes.extend([name] * w)
    if not classes:
        raise ValueError(f"empty mix weights: {weights!r}")

    out: List[str] = []
    block_idx = 0
    while len(out) < length:
        block = list(classes)
        random.Random(seed * 7919 + block_idx).shuffle(block)
        out.extend(block)
        block_idx += 1
    return out[:length]


def load_class_plan(path: str) -> List[str]:
    """Read the per-arrival class column of a dynamic trace csv, in file order.

    The file is the same canonical arrival trace the runner replays; it must be
    strictly ascending in `arrival_s` so that "file order" and the runner's
    sorted arrival order are the same sequence. The generator enforces that;
    this reader verifies it rather than trusting it, because a silent
    off-by-order here would mislabel every request's class without any error.
    """
    classes: List[str] = []
    prev = None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for need in ("arrival_s", "class"):
            if need not in cols:
                raise ValueError(
                    f"class plan {path!r} is missing the {need!r} column "
                    f"(found {cols}). Generate it with "
                    f"traces/dynamic/build_dynamic_mix_trace.py."
                )
        for lineno, row in enumerate(reader, start=2):
            t = float(row["arrival_s"])
            if prev is not None and t < prev:
                raise ValueError(
                    f"class plan {path!r} line {lineno}: arrival_s={t} goes "
                    f"backwards (previous {prev}). The runner sorts arrivals, "
                    f"so a non-ascending file would pair classes with the "
                    f"wrong requests."
                )
            prev = t
            cls = (row.get("class") or "").strip()
            if not cls:
                raise ValueError(f"class plan {path!r} line {lineno}: empty class")
            classes.append(cls)
    if not classes:
        raise ValueError(f"class plan {path!r} contains no arrivals")
    return classes


def realised_ratio(seq: List[str]) -> Dict[str, float]:
    """Fraction of each class in a sequence (for run_config bookkeeping)."""
    n = len(seq) or 1
    out: Dict[str, float] = {}
    for c in seq:
        out[c] = out.get(c, 0.0) + 1.0
    return {k: round(v / n, 4) for k, v in sorted(out.items())}
