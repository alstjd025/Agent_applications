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
    load_class_plan,
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
    """Draw tasks from per-class delegate pools following a fixed plan.

    Two plan sources (see mixplan.py):

    * `class_seq` — the static-mix cycle; draw i takes `class_seq[i % len]`.
    * `class_plan` — one entry per trace arrival, for time-varying mixes.
      Indexed by the **global** arrival index, not this worker's local one:
      `_run_multiprocess` hands worker k the arrivals `offsets[k::n]`, and its
      submit loop calls `next_task()` once per arrival in order, so worker k's
      j-th draw is global arrival `k + j*n`. Indexing by global arrival (rather
      than by wall clock) keeps the class↔time pairing exact even when the
      fleet falls behind and the open-loop driver submits back-to-back.
    """

    def __init__(self, pools: dict, class_seq: list, class_plan: Optional[list] = None,
                 shard_idx: int = 0, n_shards: int = 1):
        self.pools = pools
        self.class_seq = class_seq
        self.class_plan = class_plan
        self.shard_idx = int(shard_idx)
        self.n_shards = max(1, int(n_shards))
        self._i = 0
        self._lock = threading.Lock()

    def next_task(self) -> Optional[dict]:
        with self._lock:
            j = self._i
            self._i += 1
        if self.class_plan is not None:
            g = self.shard_idx + j * self.n_shards
            if g >= len(self.class_plan):
                # Structurally impossible (the plan file IS the trace file, so
                # arrivals and plan entries are the same rows) -- but if it ever
                # happens, say so instead of silently recycling a stale class.
                print(f"[mixed] class plan exhausted at global index {g} "
                      f"(plan has {len(self.class_plan)}); ending this shard")
                return None
            cls = self.class_plan[g]
        else:
            cls = self.class_seq[j % len(self.class_seq)]
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
        # DeadlineScheduler: {class: {"ttft_ms"|"tbt_ms"|"e2e_ms": ...}} absolute
        # per-class SLO spec, folded into `priority` by the completions client.
        self._slo = {}
        self._priority_mode = "none"
        # Dynamic-trace runs: one class per arrival, read from the trace csv.
        self._class_plan = None
        self._class_plan_file = None

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
        # A `class_plan_file` (dynamic trace) supersedes `mix`: the per-arrival
        # classes are fixed offline, so the run's weights are whatever the plan
        # actually contains. Deriving them from the plan (rather than trusting a
        # `mix` block to agree with it) keeps run_config honest and makes sure
        # every class the plan uses gets its delegate loaded.
        self._class_plan_file = workload_config.get("class_plan_file")
        if self._class_plan_file:
            self._class_plan = load_class_plan(self._class_plan_file)
            counts = {}
            for c in self._class_plan:
                counts[c] = counts.get(c, 0) + 1
            weights = counts
            print(f"[mixed] class plan {self._class_plan_file}: "
                  f"{len(self._class_plan)} arrivals, "
                  f"whole-run ratio {realised_ratio(self._class_plan)}")
        else:
            weights = dict(workload_config.get("mix", DEFAULT_MIX))
            weights = {k: int(v) for k, v in weights.items() if int(v) > 0}
        unknown = set(weights) - set(DELEGATES)
        if unknown:
            raise ValueError(f"unknown mix classes: {sorted(unknown)}")
        plan_len = int(workload_config.get("plan_length", DEFAULT_PLAN_LENGTH))
        seed = int(workload_config.get("sample_seed", args.seed))
        self._slo_budget_ms = dict(workload_config.get("slo_budget_ms", {}) or {})
        self._slo = dict(workload_config.get("slo", {}) or {})
        self._priority_mode = str(workload_config.get("priority_mode", "none"))

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

        if self._class_plan is None:
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
        shard = workload_config.get("_shard") or {}
        sidx = int(shard.get("idx", 0))
        nsh = max(1, int(shard.get("n", 1)))
        pools = {}
        for cls, ds in dataset["datasets"].items():
            sub_cfg = self._sub_config(workload_config, cls)
            # Disjoint slice per worker. run_experiment._worker_main does this
            # for workloads whose load_dataset returns a list; this one returns
            # a dict, so its isinstance check skipped us and all n workers sent
            # byte-identical prompt streams -- measured as exactly 12.00x
            # duplication of every prompt in EXP-66, at both 45 and 70 req/s.
            #
            # That inflates one quantity and no other: the share of an arriving
            # prompt the engine has to recompute. Arrival rate, class mix,
            # output lengths and SLO scoring are unaffected by a prompt being
            # sent twelve times. But prefix cache reuse is exactly the quantity
            # EXP-66 found separating llm-d from FluidServe, so every mixed-
            # workload run measured it under far more reuse than the workload
            # intends. See ms_dev/notes/fluidserve-prefix.md section 8.
            #
            # Reported rather than applied silently: a guard that skips is
            # indistinguishable from one that ran, which is how this survived.
            if isinstance(ds, list) and len(ds) >= nsh:
                ds = ds[sidx::nsh]
                print(f"[mixed] shard {sidx}/{nsh}: class {cls!r} -> "
                      f"{len(ds)} of its records")
            else:
                print(f"[mixed] shard {sidx}/{nsh}: class {cls!r} NOT sharded "
                      f"(type {type(ds).__name__}); its prompts will repeat "
                      f"across workers")
            pools[cls] = self._delegates[cls].create_task_pool(
                ds, baseline_latencies, rng, args, sub_cfg
            )
        return MixedPool(pools, self._class_seq, class_plan=self._class_plan,
                         shard_idx=shard.get("idx", 0), n_shards=shard.get("n", 1))

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
        # DeadlineScheduler: inject this class's absolute SLO spec + mode so the
        # completions client can fold it into `priority`. Only the mix adapter
        # knows the class; the delegate/engine stay class-agnostic.
        if self._priority_mode != "none":
            context = replace(
                context,
                slo_spec=self._slo.get(cls),
                priority_mode=self._priority_mode,
            )
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
        #
        # The swe figure depends on which transcript is replayed (the shortened
        # one averages 6,812 tokens against the original's 22,474), so a config
        # that changes the transcript has to state it or this bookkeeping — and
        # the input_token_fraction derived from it — would describe a different
        # workload than the one that ran.
        mean_in = {"chat": 674, "deepresearch": 4055, "swe": 22474}
        mean_in.update(workload_config.get("mean_input_tokens", {}) or {})
        total_w = sum(self._mix.values()) or 1
        req_frac = {c: round(w / total_w, 4) for c, w in sorted(self._mix.items())}
        tok_w = {c: self._mix.get(c, 0) * mean_in.get(c, 0) for c in self._mix}
        tok_total = sum(tok_w.values()) or 1
        # With a class plan the "mix" is a whole-run average of a schedule that
        # changes during the run -- label it so nobody reads it as the mix that
        # was in force at any given moment. The plan JSON has the segments.
        dyn = {}
        if self._class_plan is not None:
            dyn = {
                "class_plan_file": self._class_plan_file,
                "class_plan_arrivals": len(self._class_plan),
                "mix_is_time_varying": True,
                "note": "request_fraction/mix_weights below are WHOLE-RUN "
                        "averages of a time-varying mix; per-segment targets "
                        "and realised ratios are in the trace's .plan.json",
            }
        return {
            "name": self.name,
            **dyn,
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
                f"per-arrival plan read from {self._class_plan_file} and indexed "
                f"by GLOBAL arrival index (shard_idx + j*n_shards), so the "
                f"class<->arrival pairing is identical regardless of --load-procs"
                if self._class_plan is not None else
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
