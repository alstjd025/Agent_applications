"""k8s control for the Llumnix serving stack — per-condition cold restart.

For rigorous per-experiment isolation the user wants engine + control plane
restarted between conditions so every run starts from cold KV/prefix cache and
fresh Prometheus counters (not just window deltas). This module drives that via
kubectl, so it works both from the host (admin kubeconfig) and from an
in-cluster runner pod (ServiceAccount token + RBAC).

Facts about this deployment (deploy/neutral/full-mode-scheduling/load-balance):
  - engine  = LeaderWorkerSet `neutral`, single pod `neutral-0` (4 vLLM procs).
              LWS is a CRD this kubectl can't `rollout restart`, so we restart
              by deleting the pod; the LWS controller recreates neutral-0 with
              the same stable hostname (neutral-0.neutral.<ns>.svc). Cold start
              measured ~90s (model on local cache).
  - control = Deployments `scheduler` and `gateway` -> `kubectl rollout restart`.
  - redis (CMS store) is left running so instance discovery state is coherent.
"""

import subprocess
import time
from typing import Optional

import requests


def _kubectl(args, timeout=120, check=False):
    cp = subprocess.run(
        ["kubectl", *args], capture_output=True, text=True, timeout=timeout
    )
    if check and cp.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {cp.stderr.strip()}")
    return cp


def _pod_uid(namespace: str, pod: str) -> Optional[str]:
    cp = _kubectl(["get", "pod", pod, "-n", namespace, "-o", "jsonpath={.metadata.uid}"])
    if cp.returncode != 0:
        return None
    return cp.stdout.strip() or None


def _pod_uid_ready(namespace: str, pod: str) -> tuple:
    """(uid, ready_bool) for the named pod, or (None, None) if absent."""
    cp = _kubectl([
        "get", "pod", pod, "-n", namespace,
        "-o", "jsonpath={.metadata.uid}|{.status.conditions[?(@.type=='Ready')].status}",
    ])
    if cp.returncode != 0 or "|" not in cp.stdout:
        return (None, None)
    uid, _, ready = cp.stdout.strip().partition("|")
    return (uid or None, (ready == "True") if uid else None)


def _wait_pod_recreated_ready(
    namespace: str, pod: str, old_uid: Optional[str], timeout: int, poll: float = 3.0
) -> bool:
    """Wait until `pod` exists with a UID != old_uid and is Ready.

    Tracking the UID avoids returning early on the OLD (terminating) pod, which
    keeps the same name (LWS recreates neutral-0). old_uid=None just waits Ready.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        uid, ready = _pod_uid_ready(namespace, pod)
        if uid and (old_uid is None or uid != old_uid) and ready is True:
            return True
        time.sleep(poll)
    return False


def _wait_gateway(base_url: str, timeout: int, poll: float = 2.0) -> bool:
    """Poll {base_url}/v1/models until it serves 200 (gateway back up)."""
    deadline = time.monotonic() + timeout
    url = base_url.rstrip("/") + "/v1/models"
    while time.monotonic() < deadline:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(poll)
    return False


def restart_llumnix(
    namespace: str = "llumnix",
    engine_pod: str = "neutral-0",
    restart_control: bool = True,
    gateway_probe_url: Optional[str] = None,
    wait_timeout: int = 600,
) -> dict:
    """Cold-restart the engine (+ optionally scheduler/gateway) and wait ready.

    Returns a dict with per-phase durations and success flags. Raises only on
    kubectl invocation errors for the delete/restart trigger; readiness
    timeouts are reported as ok=False so the caller can decide.
    """
    t0 = time.monotonic()
    result = {"ok": True, "phases": {}}

    # 1) Restart the engine by deleting its pod (LWS recreates neutral-0).
    #    Use --wait=false (non-blocking): the graceful shutdown of an 8-GPU
    #    vLLM is slow/variable and would time out a blocking delete. We instead
    #    wait for a NEW pod (different UID) to become Ready in step 3.
    old_uid = _pod_uid(namespace, engine_pod)
    print(f"[llumnix-restart] deleting pod {engine_pod} -n {namespace} "
          f"(uid={old_uid}, engine cold restart)")
    _kubectl(["delete", "pod", engine_pod, "-n", namespace, "--wait=false"],
             timeout=60, check=True)

    # 2) Restart the control plane (Deployments support rollout restart).
    if restart_control:
        print("[llumnix-restart] rollout restart deploy/scheduler deploy/gateway")
        _kubectl(["rollout", "restart", "deploy/scheduler", "deploy/gateway",
                  "-n", namespace], check=True)

    # 3) Wait for a freshly-recreated engine pod to come back Ready.
    eng_ok = _wait_pod_recreated_ready(namespace, engine_pod, old_uid, wait_timeout)
    result["phases"]["engine_ready_s"] = round(time.monotonic() - t0, 1)
    result["ok"] = result["ok"] and eng_ok
    print(f"[llumnix-restart] engine ready={eng_ok} "
          f"({result['phases']['engine_ready_s']}s)")

    # 4) Wait for control-plane rollouts to finish.
    if restart_control:
        for dep in ("scheduler", "gateway"):
            cp = _kubectl(["rollout", "status", f"deploy/{dep}", "-n", namespace,
                           f"--timeout={wait_timeout}s"], timeout=wait_timeout + 30)
            ok = cp.returncode == 0
            result["ok"] = result["ok"] and ok
            print(f"[llumnix-restart] {dep} rollout ok={ok}")

    # 5) Verify the gateway actually serves again before load resumes.
    if gateway_probe_url:
        gw_ok = _wait_gateway(gateway_probe_url, wait_timeout)
        result["phases"]["gateway_serving"] = gw_ok
        result["ok"] = result["ok"] and gw_ok
        print(f"[llumnix-restart] gateway serving={gw_ok} at {gateway_probe_url}")

    result["phases"]["total_s"] = round(time.monotonic() - t0, 1)
    print(f"[llumnix-restart] done ok={result['ok']} total={result['phases']['total_s']}s")
    return result


if __name__ == "__main__":
    # Standalone host test: restart and time it. Gateway probe optional (host
    # needs a live port-forward at the given URL).
    import sys
    ns = sys.argv[1] if len(sys.argv) > 1 else "llumnix"
    probe = sys.argv[2] if len(sys.argv) > 2 else None
    r = restart_llumnix(namespace=ns, gateway_probe_url=probe)
    print(r)
