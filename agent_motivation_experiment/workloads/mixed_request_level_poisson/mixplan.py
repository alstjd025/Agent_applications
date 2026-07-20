"""Deterministic class-mixing plan for the mixed request-level workload.

The mix is specified as integer weights per class, e.g. {"chat": 1,
"deepresearch": 1, "swe": 1}. Arrivals are assigned a class by repeating a
**block** whose composition matches the weights exactly (block size =
Σweights), with the block shuffled deterministically per block index.

Why blocks instead of independent per-arrival sampling: over a 5-minute run
i.i.d. sampling leaves the realised ratio off target by a noticeable margin at
low rates (and the whole point of the experiment is to compare *ratios*).
Block-repeat pins the realised composition to the target within one block,
while the per-block shuffle keeps the classes interleaved rather than arriving
in fixed rotation (which would alias with the arrival process).
"""

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


def realised_ratio(seq: List[str]) -> Dict[str, float]:
    """Fraction of each class in a sequence (for run_config bookkeeping)."""
    n = len(seq) or 1
    out: Dict[str, float] = {}
    for c in seq:
        out[c] = out.get(c, 0.0) + 1.0
    return {k: round(v / n, 4) for k, v in sorted(out.items())}
