"""Per-run Llumnix server-side metrics collector.

Scrapes Prometheus ``/metrics`` from every layer of the Llumnix serving
stack on a fixed interval and persists one JSONL time-series file per target
under ``<run_dir>/server_metrics/``. This replaces the SGLang-specific
``parse_server_logs.py`` path (which parses SGLang stderr) — Llumnix has no
such stderr; all server-side signal lives in Prometheus endpoints.

Layers (see deploy/neutral/full-mode-scheduling/load-balance/*.yaml in the
llumnix_reproduce repo):
  - vLLM engines  : <host>:8000..8003/metrics   -> vllm:* engine metrics
  - scheduler     : <host>:8088/metrics         -> scheduler_* (rescheduling)
  - gateway       : <host>:8089/metrics         -> request_*, gateway_*,
                                                   instance_lrs_*, instance_cms_*

Design notes:
  - Reuses the daemon-thread + interval loop shape of legacy/load_monitor.py
    and the AsyncJSONLWriter background sink from metrics_tracker.py.
  - Tolerant of transient scrape failures (host-side kubectl port-forward can
    drop long-lived connections); a failed tick writes {"ok": false}.
  - There is NO migration-completion metric in this Llumnix build; the numeric
    migration signal is scheduler_rescheduling_total. Ground-truth "a KV
    transfer happened" comes from engine/scheduler logs, captured separately
    at run teardown (see capture_migration_logs()).
"""

import os
import re
import signal
import subprocess
import threading
import time
from typing import Dict, List, Optional

import requests

from metrics_tracker import AsyncJSONLWriter


# ---------------------------------------------------------------------------
# Metric name whitelists per layer (exact metric names; labels are preserved).
# Histogram families are captured via their _sum/_count so rates/averages can
# be derived across ticks in analysis. Bucket lines are intentionally skipped.
# ---------------------------------------------------------------------------
ENGINE_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:kv_cache_usage_perc",          # this build (NOT gpu_cache_usage_perc)
    "vllm:gpu_cache_usage_perc",         # fallback for older engines
    "vllm:gpu_cache_usage_blocks",
    "vllm:gpu_cache_total_blocks",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:request_success_total",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_queue_time_seconds_count",     # +count -> avg queueing time
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "vllm:prefix_cache_queries_total",           # KV/prefix cache hit rate =
    "vllm:prefix_cache_hits_total",              #   hits/queries
}

SCHEDULER_METRICS = {
    "scheduler_scheduling_total",
    "scheduler_scheduling_failed_total",
    "scheduler_rescheduling_total",          # migration-decision counter
    "scheduler_rescheduling_failed_total",
    "scheduler_cms_refresh_metadata_duration_milliseconds_sum",
    "scheduler_cms_refresh_status_duration_milliseconds_sum",
    # per-instance CMS phase split (exposed on the SCHEDULER endpoint, keyed by
    # instance_id) -> prefill vs decode request counts, projected KV load signal.
    "instance_cms_running_requests",
    "instance_cms_waiting_requests",
    "instance_cms_decode_batch_size",
    "instance_cms_all_prefills_tokens_num",
    "instance_cms_all_decodes_tokens_num",
    "instance_cms_kv_cache_usage_ratio_projected",
    "instance_cms_inflight_dispatch_prefill_requests",
    "instance_cms_inflight_dispatch_decode_requests",
    "instance_cms_scheduler_waiting_to_decode_requests",
    # PolyServe tier repartitioning (labelled by tpot_slo_ms). The scheduler
    # also klog's each decision, but a long dynamic-trace run outlives the pod
    # log buffer, so the allocation history has to come from the scrape.
    "scheduler_polyserve_tier_servers",
    "scheduler_polyserve_tier_demand",
    "scheduler_polyserve_live_servers",
    # FluidServe. Same reason as above: the per-decision detail is only in the
    # scheduler's log, which does not survive an hour-long run. What the run has
    # to be able to answer afterwards is why each request went where it did, so
    # the per-instance state at decision time is exported.
    "scheduler_fluidserve_decisions_total",       # route / pend / shed / force
    "scheduler_fluidserve_headroom_tokens",       # per instance
    "scheduler_fluidserve_cap_kv_tokens",         # latency-imposed capacity
    "scheduler_fluidserve_projected_kv_tokens",
    "scheduler_fluidserve_outflow_tokens",
    "scheduler_fluidserve_observed_step_ms",      # measured iteration time
    "scheduler_fluidserve_predicted_step_ms",     # what the model said
    "scheduler_fluidserve_tightest_allowance_ms",  # tightest remaining budget
    "scheduler_fluidserve_gate_allowance_ms",      # the nominal pace admission is gated on
    "scheduler_fluidserve_live_requests",
    "scheduler_fluidserve_unachievable_requests",
    "scheduler_fluidserve_retired_total",
    "scheduler_fluidserve_capacity_correction",   # measured mean step / predicted
    "scheduler_fluidserve_prefill_fraction",      # measured share of a prompt actually computed
    "scheduler_fluidserve_arriving_prefill_tokens",  # projected over the horizon
    "scheduler_fluidserve_queued_prefill_tokens",    # what the engine reports right now
    # EXP-47. Whether the second placement inside one engine status step sees
    # the first. §40.3 measured an engine taking 3.2x its own headroom in a
    # minute and §41 measured 50.4% of 500 ms steps carrying more than one
    # placement; §42 could not settle it from a 1 Hz gauge because the quantity
    # is per DECISION, not per scrape.
    # A histogram is exposed under THREE names -- _sum, _count and _bucket -- and
    # this allowlist matches exactly (`if name not in wanted`), so listing the
    # base name collects nothing. Every other histogram family here is listed as
    # an explicit _sum/_count pair for that reason; these were added as base
    # names on 2026-08-01 and the first EXP-47 run collected neither.
    "scheduler_fluidserve_dispatch_ordinal_in_step_sum",
    "scheduler_fluidserve_dispatch_ordinal_in_step_count",
    "scheduler_fluidserve_headroom_move_in_step_sum",
    "scheduler_fluidserve_headroom_move_in_step_count",
    # Same omission, older: these three are observed in commit() and have never
    # been collected. headroom_at_dispatch is what §40.3's "3.2 times its own
    # headroom" question is asked of directly.
    "scheduler_fluidserve_headroom_at_dispatch_sum",
    "scheduler_fluidserve_headroom_at_dispatch_count",
    "scheduler_fluidserve_harm_sum",
    "scheduler_fluidserve_harm_count",
    "scheduler_fluidserve_class_share_sum",
    "scheduler_fluidserve_class_share_count",
    "scheduler_fluidserve_prefill_duty",          # per-instance share of engine time on
                                                  # prefill; what the projection is built
                                                  # from since v19
    # The terms the predicted iteration is assembled from, added for EXP-29.
    # observed_step_ms and predicted_step_ms above give the two endpoints of the
    # gap that holds 84% of decisions at 80 req/s, and a gap between them can be
    # produced at four different places with no way to tell which from outside.
    # These are the places.
    "scheduler_fluidserve_decode_only_ms",    # decode law at the batch that RAN, which is
                                              # what the prefill attribution subtracts
    "scheduler_fluidserve_decode_law_ms",     # decode law at the batch the PREDICTION uses
    "scheduler_fluidserve_obs_kv_tokens",     # the KV the first of those was evaluated at
    "scheduler_fluidserve_obs_decode_batch",  # and the request count
    "scheduler_fluidserve_pace_ms",           # iteration time the horizon is converted with;
                                              # enters the prefill term as duty x pace
    "scheduler_fluidserve_obs_steps",         # iterations the measured interval covered. The
                                              # per-interval mean is elapsed/steps, so without
                                              # this the series cannot be aggregated into a
                                              # time per TOKEN -- and the equal-weight average
                                              # of it reads 16 ms above the engines' own
                                              # inter-token latency.
    # What canWait reserves for everything that happens after a placement, and the
    # mean it is bounded from. The gap between them is the whole of the v27 change.
    "scheduler_fluidserve_placement_delay_bound_ms",
    "scheduler_fluidserve_placement_delay_mean_ms",
    # How often the KV projection changed a decision, counted inside the run
    # rather than inferred from an ablation arm whose effect has to survive the
    # feedback the change itself causes.
    "scheduler_fluidserve_flux_flips_total",
    "scheduler_fluidserve_flux_evaluations_total",
    "scheduler_fluidserve_raw_step_ms",       # that interval's elapsed/steps, before the
                                              # step-weighted smoothing observed_step_ms now
                                              # carries
    "scheduler_fluidserve_offered_rate_tokens_per_ms",  # telemetry only since v19: how far
                                                  # past the fleet's capacity the run was
                                                  # driven. No decision reads it.
    # How long one scheduling call takes. FluidServe's decision is far heavier
    # than a filter-and-pick, and a request held at the gateway re-enters this
    # path at every retry, so the per-call cost and the call rate together
    # decide whether the control plane can keep up with the offered load.
    "request_full_mode_schedule_duration_milliseconds_sum",
    "request_full_mode_schedule_duration_milliseconds_count",
    "scheduler_scheduling_total",
    "scheduler_scheduling_failed_total",
}

GATEWAY_METRICS = {
    # How often, and for how long, the gateway held a request because the
    # scheduler had nowhere to put it yet. Under FluidServe this is not an error
    # path but the mechanism by which a placement is deferred, so it is the
    # counterpart of the scheduler's pend decisions.
    "gateway_scheduling_waited_total",
    "gateway_scheduling_wait_milliseconds_sum",
    "gateway_scheduling_wait_milliseconds_count",
    "gateway_scheduling_gave_up_total",
    "gateway_scheduling_rejected_total",
    "gateway_pending_requests",
    "gateway_current_requests",
    "request_total",
    "request_retry_total",
    "request_fallback_total",
    "request_input_tokens_total",
    "request_output_tokens_total",
    "request_e2e_latency_seconds_sum",
    "request_e2e_latency_seconds_count",
    "request_ttft_milliseconds_sum",
    "request_ttft_milliseconds_count",
    "request_tpot_milliseconds_sum",
    "request_tpot_milliseconds_count",
    "request_queue_duration_milliseconds_sum",
    "request_schedule_duration_milliseconds_sum",
    # per-instance load (gateway realtime + CMS-from-redis)
    "instance_lrs_running_requests",
    "instance_lrs_waiting_requests",
    "instance_lrs_running_tokens",
    "instance_lrs_total_requests",
    "instance_cms_running_requests",
    "instance_cms_waiting_requests",
    "instance_cms_used_gpu_tokens",
    "instance_cms_kv_cache_usage_ratio_projected",   # the rescheduler's load signal
    "instance_cms_decode_batch_size",
}


# ---------------------------------------------------------------------------
# Prometheus text parsing
# ---------------------------------------------------------------------------
_SERIES_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"      # metric name
    r"(?:\{(?P<labels>[^}]*)\})?"                 # optional {labels}
    r"\s+(?P<value>[-+0-9.eEnaN]+)"               # value (incl. NaN/inf-ish)
    r"(?:\s+\d+)?\s*$"                            # optional timestamp
)


def parse_prometheus(text: str, wanted: set) -> Dict[str, float]:
    """Parse Prometheus exposition text into a flat {series_key: value} dict.

    series_key = metric name, or ``name|k=v,k=v`` when labels are present, so
    per-instance/per-model series stay distinguishable in one flat record.
    Only metric names in ``wanted`` are kept; bucket lines and comments skip.
    """
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _SERIES_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if name not in wanted:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        labels = m.group("labels")
        if labels:
            # normalise: strip quotes, sort by key for a stable series_key
            parts = []
            for kv in labels.split(","):
                if "=" not in kv:
                    continue
                k, _, v = kv.partition("=")
                parts.append(f"{k.strip()}={v.strip().strip(chr(34))}")
            key = name + "|" + ",".join(sorted(parts)) if parts else name
        else:
            key = name
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Target descriptor
# ---------------------------------------------------------------------------
class ScrapeTarget:
    def __init__(self, label: str, url: str, wanted: set):
        self.label = label          # file name stem, e.g. "engine_8000"
        self.url = url              # http://host:port/metrics
        self.wanted = wanted


def default_llumnix_targets(
    engine_host: str = "localhost",
    scheduler_host: str = "localhost",
    gateway_host: str = "localhost",
    engine_ports=(8000, 8001, 8002, 8003),
    scheduler_port: int = 8088,
    gateway_port: int = 8089,
) -> List[ScrapeTarget]:
    """Build the canonical target list for the 4×TP2 neutral deployment.

    Hosts are per-layer so this works both from the host via a single
    port-forward (all three = localhost, different ports) and from an
    in-cluster runner pod via k8s DNS (engine=neutral-0.neutral,
    scheduler=scheduler, gateway=gateway — restart-stable names).
    """
    targets: List[ScrapeTarget] = []
    for p in engine_ports:
        targets.append(ScrapeTarget(f"engine_{p}", f"http://{engine_host}:{p}/metrics", ENGINE_METRICS))
    targets.append(ScrapeTarget("scheduler", f"http://{scheduler_host}:{scheduler_port}/metrics", SCHEDULER_METRICS))
    targets.append(ScrapeTarget("gateway", f"http://{gateway_host}:{gateway_port}/metrics", GATEWAY_METRICS))
    return targets


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------
class LlumnixMetricsCollector:
    """Background time-series scraper for the Llumnix serving stack.

    One daemon thread scrapes every target sequentially each ``interval``
    seconds and appends a timestamped record to a per-target JSONL file via
    AsyncJSONLWriter. Start once at run start, stop at run teardown.
    """

    def __init__(self, targets: List[ScrapeTarget], out_dir: str, interval: float = 1.0,
                 capture_dispatch: bool = True, namespace: str = "llumnix"):
        self.targets = targets
        self.out_dir = out_dir
        self.interval = interval
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._writers: Dict[str, AsyncJSONLWriter] = {
            t.label: AsyncJSONLWriter(f"{out_dir.rstrip('/')}/{t.label}.jsonl")
            for t in targets
        }
        # Scheduler dispatch-log streaming (per-request -> engine attribution)
        self.capture_dispatch = capture_dispatch
        self.namespace = namespace
        self._dispatch_proc = None
        self._dispatch_fh = None

    def _start_dispatch_capture(self):
        """Stream the scheduler's per-request dispatch decisions to a file.

        The scheduler logs "[Schedule] dispatch request <uuid> to neutral
        instance <instance_id>" per request and periodic
        "[refreshInstanceMetadata] instanceID=.. api_server_port:80xx" lines.
        Together they map a request to the engine that served it (joined with
        the client-side request_ids.jsonl sidecar). The raw log is dominated by
        per-tick filter spam, so we grep at the source and keep only the lines
        the join needs. Best-effort: failures never break the run.
        See experiments/DEV_request-engine-attribution.md.
        """
        if not self.capture_dispatch:
            return
        path = f"{self.out_dir.rstrip('/')}/scheduler_dispatch.log"
        pattern = (r"\[Schedule\] dispatch request|refreshInstanceMetadata|"
                   r"Generate rescheduling pairs|Received Migration")
        # Follow the NEWEST RUNNING scheduler pod, re-resolved on every attach.
        # `kubectl logs -f deploy/scheduler` binds to whichever pod it picks at
        # attach time; right after --restart-per-condition's rollout that can be
        # the *terminating* old pod, which still emits periodic metadata lines
        # but receives no traffic — producing a dispatch log with 0 dispatch
        # lines (observed, EXP-14 mix smoke). The retry loop also survives pod
        # churn mid-run: if the stream ends we re-resolve and re-attach.
        script = (
            f"while :; do "
            f"  POD=$(kubectl -n {self.namespace} get pods -l app=scheduler "
            f"        --sort-by=.metadata.creationTimestamp "
            f"        -o jsonpath='{{range .items[?(@.status.phase==\"Running\")]}}"
            f"{{.metadata.name}}{{\"\\n\"}}{{end}}' 2>/dev/null | tail -1); "
            f"  [ -n \"$POD\" ] || {{ sleep 2; continue; }}; "
            f"  kubectl logs -n {self.namespace} -f --tail=0 \"$POD\" 2>/dev/null "
            f"    | grep -E --line-buffered '{pattern}'; "
            f"  sleep 1; "
            f"done"
        )
        try:
            self._dispatch_fh = open(path, "w", encoding="utf-8")
            self._dispatch_proc = subprocess.Popen(
                ["sh", "-c", script],
                stdout=self._dispatch_fh, stderr=subprocess.DEVNULL,
                start_new_session=True,   # so we can kill the whole pipeline
            )
            print(f"[LlumnixMetrics] dispatch-log capture -> {path}")
        except Exception as e:
            print(f"[LlumnixMetrics] dispatch-log capture failed to start: {e}")
            self._dispatch_proc = None

    def _stop_dispatch_capture(self):
        if self._dispatch_proc is not None:
            # The capture is a `while` loop wrapping a kubectl|grep pipeline, so
            # terminating just the shell would orphan the children. It runs in
            # its own session (start_new_session=True) -> kill the whole group.
            try:
                os.killpg(os.getpgid(self._dispatch_proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    self._dispatch_proc.terminate()
                except Exception:
                    pass
            try:
                self._dispatch_proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._dispatch_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self._dispatch_proc = None
        if self._dispatch_fh is not None:
            try:
                self._dispatch_fh.close()
            except Exception:
                pass
            self._dispatch_fh = None

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._start_dispatch_capture()
        print(
            f"[LlumnixMetrics] Started (interval={self.interval}s, "
            f"{len(self.targets)} targets -> {self.out_dir}/)"
        )

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=self.interval + 5)
        for w in self._writers.values():
            w.close()
        self._stop_dispatch_capture()
        print("[LlumnixMetrics] Stopped")

    def _loop(self):
        while self.is_running:
            tick = time.time()
            for t in self.targets:
                record = {"t": tick, "ok": True}
                try:
                    resp = requests.get(t.url, timeout=2)
                    resp.raise_for_status()
                    record.update(parse_prometheus(resp.text, t.wanted))
                except Exception:
                    record["ok"] = False
                self._writers[t.label].write(record)
            # keep cadence roughly fixed regardless of scrape duration
            elapsed = time.time() - tick
            time.sleep(max(0.0, self.interval - elapsed))


# ---------------------------------------------------------------------------
# Migration ground-truth log capture (no completion metric exists)
# ---------------------------------------------------------------------------
def capture_migration_logs(
    out_path: str,
    scheduler_selector: str = "app=scheduler",
    engine_selector: str = "llumnix.io/infer-type=neutral",
    namespace: str = "llumnix",
    tail: int = 5000,
) -> bool:
    """Grep scheduler + engine pod logs for migration/rescheduling evidence.

    Writes matching lines to ``out_path``. Returns False if kubectl is
    unavailable or both greps fail. Best-effort: this is the only ground-truth
    that a KV transfer actually happened (scheduler_rescheduling_total only
    counts decisions).
    """
    patterns = "rescheduling|Generate rescheduling pairs|Received Migration|" \
               "MigrationFrontend|No requests to migrate|migrate"
    wrote_any = False
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for title, selector in (
                ("scheduler", scheduler_selector),
                ("engine(neutral)", engine_selector),
            ):
                f.write(f"\n===== {title} ({selector}) =====\n")
                try:
                    logs = subprocess.run(
                        ["kubectl", "logs", "-n", namespace, "-l", selector,
                         "--tail", str(tail), "--prefix", "--timestamps"],
                        capture_output=True, text=True, timeout=60,
                    )
                    if logs.returncode != 0:
                        f.write(f"[kubectl logs failed: {logs.stderr.strip()}]\n")
                        continue
                    matched = [
                        ln for ln in logs.stdout.splitlines()
                        if re.search(patterns, ln, re.IGNORECASE)
                    ]
                    f.write("\n".join(matched) + ("\n" if matched else "[no matching lines]\n"))
                    wrote_any = wrote_any or bool(matched)
                except Exception as e:
                    f.write(f"[error: {e}]\n")
    except Exception as e:
        print(f"[LlumnixMetrics] capture_migration_logs failed: {e}")
        return False
    return wrote_any


if __name__ == "__main__":
    # Smoke self-test against a locally port-forwarded stack.
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/llumnix_metrics_selftest"
    import os
    os.makedirs(out, exist_ok=True)
    c = LlumnixMetricsCollector(default_llumnix_targets(), out_dir=out, interval=1.0)
    c.start()
    time.sleep(5)
    c.stop()
    print(f"wrote sample series to {out}/")
