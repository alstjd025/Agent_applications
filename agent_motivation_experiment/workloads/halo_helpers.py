"""Shared client-side helpers for Project Halo (request-level).

See ms_dev/halo_dev/halo_api_reference.md in the sglang repo for the
authoritative API spec. Halo was refactored job-level -> request-level on
2026-05-19: there is no job pre-registration anymore. Admission is fully
per-request, and clients only need to:

  1. probe GET /halo/status once at startup (mode-match check), and
  2. attach the per-request SLO fields (halo_ttft_slo / halo_tbt_slo /
     halo_e2e_slo) to every generation request body via extra_body.

This module provides the startup probe. The per-request SLO wiring lives
in workloads/swe_bench_coding/agent.py::make_llm.

Failure policy:
  - mismatch between client --halo-enabled and the server's Halo state ->
    HaloConfigError, run aborts.
  - server unreachable / non-200 / malformed status -> HaloConfigError.

When `context.halo_enabled is False`, the probe still runs (to catch a
server that has Halo ON while the client doesn't) but no SLO fields are
attached to requests.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import requests  # already used elsewhere in run_experiment.py

logger = logging.getLogger(__name__)

# The probe runs once at run start so 5s is plenty.
_PROBE_TIMEOUT_S = 5.0


class HaloConfigError(RuntimeError):
    """Server-side Halo configuration doesn't match client expectations."""


def probe_halo_status(base_url: str) -> Dict[str, Any]:
    """GET /halo/status on the running sglang server.

    Returns the parsed body dict on 200. The request-level status body is:
        {"enabled": bool, "admission_policy": str, "slo_mode": str,
         "kv_cap_ratio": float, "tick_interval_ms": float}

    Raises HaloConfigError on any failure (non-200, network error, missing
    'enabled' field).
    """
    url = base_url.rstrip("/") + "/halo/status"
    try:
        r = requests.get(url, timeout=_PROBE_TIMEOUT_S)
    except requests.RequestException as e:
        raise HaloConfigError(
            f"halo status probe failed: GET {url} -> {e}"
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
    """Probe + assert mismatch. Always returns the server status dict so
    the caller can record it in run_meta.json / print it.

    - client wants Halo + server has it off -> abort
    - client wants no Halo + server has it on -> also abort (the server
      may shed the client's traffic via admission control and the client
      would never attach SLO fields, masking the rejections)
    """
    status = probe_halo_status(base_url)
    server_on = bool(status.get("enabled"))
    if client_wants_halo and not server_on:
        raise HaloConfigError(
            "client passed --halo-enabled but the sglang server has Halo "
            "OFF. Relaunch sglang with --halo-enabled (and an "
            "--halo-admission-policy) before running this experiment."
        )
    if (not client_wants_halo) and server_on:
        raise HaloConfigError(
            "client did NOT pass --halo-enabled but the sglang server "
            "has Halo ON. Either pass --halo-enabled or relaunch the "
            "server without Halo."
        )
    return status
