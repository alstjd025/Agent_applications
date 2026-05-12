"""Shared client-side helpers for Project Halo Phase 1.

See ms_dev/halo_dev/halo_api_reference.md in the sglang repo for the
authoritative API spec. This module is the *client* counterpart that
every workload's run_job() uses.

Two helpers, both used by all workloads:
  - probe_halo_status(base_url) — readiness-time check that the server
    has Halo enabled. Used by run_experiment.py before traffic starts.
  - register_halo_program(...) — chain-start one-shot POST that
    pre-registers a job (Option A) so subsequent LLM requests can pass
    halo_job_id without being rejected.

Failure policy (per C1/C7 from ms_dev/halo_dev/CLAUDE.md §13 decisions):
  - mismatch between client --halo-enabled and server's Halo state →
    HaloConfigError, run aborts.
  - 409 duplicate job_id at register time → HaloRegisterError, run
    aborts (per C2: never expected in normal operation).
  - server reachable but Halo off while client thinks it's on →
    HaloConfigError, run aborts.
  - network error during probe → HaloConfigError, run aborts.

When `context.halo_enabled is False`, none of these helpers should ever
be called by a workload — the wiring is gated at the run_job entry
point. So this module never silently no-ops.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests  # already used elsewhere in run_experiment.py

logger = logging.getLogger(__name__)

# HTTP timeouts. The probe runs once at run start so 5s is plenty;
# register runs at every job's chain start so we keep it tight to fail
# fast on misconfiguration but generous enough to absorb a transient
# TCP hiccup.
_PROBE_TIMEOUT_S = 5.0
_REGISTER_TIMEOUT_S = 5.0


class HaloConfigError(RuntimeError):
    """Server-side Halo configuration doesn't match client expectations."""


class HaloRegisterError(RuntimeError):
    """register_halo_program failed (duplicate, malformed, etc.)."""


def probe_halo_status(base_url: str) -> Dict[str, Any]:
    """GET /halo/status on the running sglang server.

    Returns the parsed body dict on 200. Raises HaloConfigError on any
    failure (non-200, network error, missing fields). The caller in
    run_experiment.py uses this once during the readiness phase to
    abort early when the client wants Halo on but the server has it
    off, and vice versa.
    """
    url = base_url.rstrip("/") + "/halo/status"
    try:
        r = requests.get(url, timeout=_PROBE_TIMEOUT_S)
    except requests.RequestException as e:
        raise HaloConfigError(
            f"halo status probe failed: GET {url} → {e}"
        ) from e
    if r.status_code != 200:
        raise HaloConfigError(
            f"halo status probe returned HTTP {r.status_code}: {r.text[:200]}"
        )
    try:
        body = r.json()
    except ValueError as e:
        raise HaloConfigError(
            f"halo status probe returned non-JSON body: {r.text[:200]}"
        ) from e
    if "enabled" not in body:
        raise HaloConfigError(
            f"halo status probe body missing 'enabled' field: {body!r}"
        )
    return body


def assert_halo_mode_matches(
    base_url: str, client_wants_halo: bool
) -> Dict[str, Any]:
    """Probe + assert mismatch. Always returns the server status dict
    so the caller can record it in run_meta.json.

    - client wants Halo + server has it off → abort
    - client wants no Halo + server has it on → also abort (LLM calls
      would need halo_job_id and we wouldn't send it)
    """
    status = probe_halo_status(base_url)
    server_on = bool(status.get("enabled"))
    if client_wants_halo and not server_on:
        raise HaloConfigError(
            "client passed --halo-enabled but the sglang server has Halo "
            "OFF. Start sglang with --halo-enabled (or "
            "`source ms_dev/experiments/halo_observe_only.sh`) before "
            "running this experiment."
        )
    if (not client_wants_halo) and server_on:
        raise HaloConfigError(
            "client did NOT pass --halo-enabled but the sglang server "
            "has Halo ON. Either pass --halo-enabled or restart the "
            "server without Halo (sglang would reject every request "
            "with HALO_NO_JOB_ID)."
        )
    return status


def register_halo_program(
    base_url: str,
    *,
    job_id: str,
    slo: float,
    total_calls: Optional[int] = None,
    stage_sequence: Optional[List[str]] = None,
    expected_input_lens: Optional[List[int]] = None,
    expected_output_lens: Optional[List[int]] = None,
    dag: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """POST /halo/programs for one job. Called at chain-start.

    Returns the server's response body dict (which carries
    {registered: True, active_jobs: N} on success).

    Raises HaloRegisterError on any non-200 — including 409 duplicate
    job_id (per C2). The 409 path is "should never happen" in normal
    operation; if it does, the user wants to know.
    """
    url = base_url.rstrip("/") + "/halo/programs"
    payload: Dict[str, Any] = {"job_id": job_id, "slo": slo}
    if total_calls is not None:
        payload["total_calls"] = total_calls
    if stage_sequence is not None:
        payload["stage_sequence"] = stage_sequence
    if expected_input_lens is not None:
        payload["expected_input_lens"] = expected_input_lens
    if expected_output_lens is not None:
        payload["expected_output_lens"] = expected_output_lens
    if dag is not None:
        payload["dag"] = dag

    try:
        r = requests.post(url, json=payload, timeout=_REGISTER_TIMEOUT_S)
    except requests.RequestException as e:
        raise HaloRegisterError(
            f"halo register failed: POST {url} job_id={job_id} → {e}"
        ) from e
    if r.status_code == 200:
        try:
            return r.json()
        except ValueError:
            return {"registered": True, "job_id": job_id}
    # 409 / 400 — surface verbatim. Caller aborts.
    raise HaloRegisterError(
        f"halo register failed for job_id={job_id}: HTTP {r.status_code} "
        f"body={r.text[:300]}"
    )
