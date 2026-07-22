"""Mixed request-level Poisson workload (chat + deep-research + SWE).

One flat open-loop Poisson stream whose arrivals are drawn from **three**
request-level workloads in a configured ratio:

| class          | delegate workload                     | input mean |
|----------------|---------------------------------------|-----------|
| `chat`         | `sharegpt_request_level_poisson`      | ~0.7k tok |
| `deepresearch` | `searcharena_request_level_poisson`   | ~4.1k tok |
| `swe`          | `codingagent_request_level_poisson`   | ~22k tok  |

This adapter owns no request content of its own: it holds one instance of
each delegate, asks each for its dataset/task pool, and dispatches
`run_job` to the delegate that owns the drawn task. So every class keeps
its exact single-workload semantics (same prompts, same determinism, same
absolute-SLO/no-abort contract), and a mixed run is directly comparable to
the single-workload baselines (EXP-12 chat, EXP-13 deep-research, EXP-10
SWE).

**No metrics schema change.** Rows are written by the delegates exactly as
in a single-workload run (`agent == "request"`). The class of each row is
recoverable from the `task_id` prefix — `sg-` chat, `sa-` deep-research,
SWE transcript ids otherwise — so existing analysis scripts keep working
unchanged and per-class breakdowns are a groupby away.

Mixing is by **request count** (not token mass): weights {1,1,1} means one
third of *arrivals* per class. Because the classes' input sizes differ by
~30x, the resulting *token* mix is very different from the request mix —
`metadata()` records both so the distinction is explicit in run_config.json.

See `mixplan.py` for how arrivals are assigned to classes deterministically.
"""

import threading
from dataclasses import replace
from typing import Optional

from workloads.base import JobResult, RunContext, TaskLogInfo
from workloads.mixed_request_level_poisson.mixplan import (
    build_class_sequence,
    realised_ratio,
)

# class label -> delegate workload module name
DELEGATES = {
    "chat": "sharegpt_request_level_poisson",
    "deepresearch": "searcharena_request_level_poisson",
    "swe": "codingagent_request_level_poisson",
}
DEFAULT_MIX = {"chat": 1, "deepresearch": 1, "swe": 1}
# How many arrivals to pre-plan. Longer than any run needs; the pool cycles.
DEFAULT_PLAN_LENGTH = 200000


class MixedPool:
    """Draw tasks from per-class delegate pools following a fixed plan."""

    def __init__(self, pools: dict, class_seq: list):
        self.pools = pools
        self.class_seq = class_seq
        self._i = 0
        self._lock = threading.Lock()

    def next_task(self) -> Optional[dict]:
        with self._lock:
            cls = self.class_seq[self._i % len(self.class_seq)]
            self._i += 1
        task = self.pools[cls].next_task()
        if task is None:
            return None
        # Tag the class so run_job knows which delegate owns this task.
        # (Analysis does not depend on this key — it uses the task_id prefix.)
        task["_mix_class"] = cls
        return task


class Workload:
    name = "mixed_request_level_poisson"

    def __init__(self):
        self._delegates = {}
        self._mix = dict(DEFAULT_MIX)
        self._class_seq = []
        self._per_class_meta = {}
        # EDF only: {class: SLO budget ms}; empty -> no priority sent.
        self._slo_budget_ms = {}

    # -- helpers ---------------------------------------------------------
    def _load_delegates(self, weights):
        from workloads import load_workload

        for cls, mod in DELEGATES.items():
            if int(weights.get(cls, 0)) > 0 and cls not in self._delegates:
                self._delegates[cls] = load_workload(mod)

    @staticmethod
    def _sub_config(workload_config: dict, cls: str) -> dict:
        """Per-class delegate config, e.g. {"deepresearch_config": {...}}."""
        return dict(workload_config.get(f"{cls}_config", {}) or {})

    # -- adapter contract ------------------------------------------------
    def load_dataset(self, args, workload_config: dict):
        weights = dict(workload_config.get("mix", DEFAULT_MIX))
        weights = {k: int(v) for k, v in weights.items() if int(v) > 0}
        unknown = set(weights) - set(DELEGATES)
        if unknown:
            raise ValueError(f"unknown mix classes: {sorted(unknown)}")
        plan_len = int(workload_config.get("plan_length", DEFAULT_PLAN_LENGTH))
        seed = int(workload_config.get("sample_seed", args.seed))
        self._slo_budget_ms = dict(workload_config.get("slo_budget_ms", {}) or {})

        self._mix = weights
        self._load_delegates(weights)

        datasets = {}
        for cls in weights:
            sub_cfg = self._sub_config(workload_config, cls)
            print(f"[mixed] loading delegate '{cls}' ({DELEGATES[cls]}) ...")
            datasets[cls] = self._delegates[cls].load_dataset(args, sub_cfg)
            try:
                self._per_class_meta[cls] = self._delegates[cls].metadata(args, sub_cfg)
            except Exception:
                self._per_class_meta[cls] = {}

        self._class_seq = build_class_sequence(weights, plan_len, seed)
        print(f"[mixed] mix={weights} -> realised request ratio "
              f"{realised_ratio(self._class_seq)} over {plan_len} planned arrivals")
        return {"datasets": datasets, "weights": weights, "seed": seed}

    def build_baseline_tasks(self, dataset, replay_count, rng, args, workload_config):
        """Concurrency-1 pass: every class's baseline tasks, concatenated."""
        tasks = []
        for cls, ds in dataset["datasets"].items():
            sub_cfg = self._sub_config(workload_config, cls)
            sub = self._delegates[cls].build_baseline_tasks(
                ds, replay_count, rng, args, sub_cfg
            )
            for t in sub:
                t["_mix_class"] = cls
            tasks.extend(sub)
        return tasks

    def create_task_pool(self, dataset, baseline_latencies, rng, args, workload_config):
        pools = {}
        for cls, ds in dataset["datasets"].items():
            sub_cfg = self._sub_config(workload_config, cls)
            pools[cls] = self._delegates[cls].create_task_pool(
                ds, baseline_latencies, rng, args, sub_cfg
            )
        return MixedPool(pools, self._class_seq)

    def run_job(self, task: dict, context: RunContext) -> JobResult:
        cls = task.get("_mix_class")
        delegate = self._delegates.get(cls)
        if delegate is None:
            raise RuntimeError(f"no delegate loaded for mix class {cls!r}")
        # EDF (EXP-15): give this request its class's SLO budget so the
        # completions client can stamp an absolute deadline as `priority`.
        # Only the mix adapter knows the class, so the substitution happens
        # here; the delegate and the engine stay class-agnostic. Absent
        # config -> None -> no priority sent (FIFO/SJF/SRPF).
        budget = self._slo_budget_ms.get(cls) if self._slo_budget_ms else None
        if budget is not None:
            context = replace(context, slo_budget_ms=int(budget))
        return delegate.run_job(task, context)

    def task_log_info(self, task: dict) -> TaskLogInfo:
        cls = task.get("_mix_class")
        delegate = self._delegates.get(cls)
        if delegate is None:
            return TaskLogInfo(task_id=task.get("instance_id", ""),
                               problem_statement="", repo="")
        info = delegate.task_log_info(task)
        return TaskLogInfo(task_id=info.task_id,
                           problem_statement=f"[{cls}] {info.problem_statement}",
                           repo=info.repo)

    def metadata(self, args, workload_config: dict) -> dict:
        # Report the request mix AND the token mix it implies — with ~30x
        # input-size spread between classes these are very different, and
        # conflating them is the easiest way to misread a mixed run.
        mean_in = {"chat": 674, "deepresearch": 4055, "swe": 22474}
        total_w = sum(self._mix.values()) or 1
        req_frac = {c: round(w / total_w, 4) for c, w in sorted(self._mix.items())}
        tok_w = {c: self._mix.get(c, 0) * mean_in.get(c, 0) for c in self._mix}
        tok_total = sum(tok_w.values()) or 1
        return {
            "name": self.name,
            "mix_weights": self._mix,
            "request_fraction": req_frac,
            "input_token_fraction_expected": {
                c: round(v / tok_total, 4) for c, v in sorted(tok_w.items())
            },
            "mean_input_tokens_assumed": mean_in,
            "delegates": {c: DELEGATES[c] for c in self._mix},
            "per_class_metadata": self._per_class_meta,
            "arrival": "request-level Poisson (lambda = requests/sec), classes "
                       "drawn by shuffled fixed-composition blocks",
            "goodput_model": "absolute SLO thresholds (no baseline, tau unused)",
            "class_from": "task_id prefix (sg-=chat, sa-=deepresearch, else swe)",
        }

    def reproducibility_config(self, args, workload_config: dict) -> dict:
        return {
            "client_seed": args.seed,
            "mix_weights": self._mix,
            "class_sequence": (
                "shuffled blocks of size sum(weights); block b shuffled with "
                "random.Random(sample_seed*7919+b) -> exact ratio per block, "
                "deterministic across runs and load processes"
            ),
            "per_class_reproducibility": {
                c: self._delegates[c].reproducibility_config(
                    args, self._sub_config(workload_config, c))
                for c in self._mix if c in self._delegates
            },
            "aborts": "all client-side aborts disabled (inherited from delegates)",
        }
