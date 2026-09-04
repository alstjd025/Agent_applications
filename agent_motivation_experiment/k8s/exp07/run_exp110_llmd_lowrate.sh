#!/usr/bin/env bash
# EXP-66 — llm-d predicted-latency (arm llmd-slo) on EXP-53's rate sweep.
#
# Design and judgement rules: llumnix_reproduce/ms_dev/notes/llmd-baseline.md
# and experiments/EXP-66_llm-d-baseline.md. This script only executes them.
#
# One condition is:
#   1. restart the engine and the three pods that hold predictor state
#   2. check the training server really came back with zero samples
#   3. pre-run 3 minutes at the condition's own rate so the predictor trains
#   4. wait until all four engines are idle
#   5. measure 8 minutes (the runner adds its usual 60 s warmup, as EXP-53 did)
#   6. copy the Envoy access log slice and the EPP metrics into the result dir
#
# Why the driver owns the restarts rather than --restart-per-condition: the
# runner's restart helper knows about deploy/scheduler and deploy/gateway, and
# neither is in this path. What has to come back, and in this order, is the
# engine and then the predictor trio -- and "the predictor" is three pods, not
# one. If any of the three keeps state across conditions the model is better at
# the later, higher rates and the rate curve itself is wrong.
#
# EXP-110 (2026-09-02). A snapshot of the EXP-108 driver that re-measures the
# ONE arrival rate whose two repeats disagreed: 10 req/s, where the rejection
# rate came out 0.0% in repeat 1 and 39.9% in repeat 2 on the same workload with
# the same request count and all four engines healthy. EXP-108 section 8.4 could
# not say why, because the training server's log was saved with --tail=2000 and
# that log runs at about 25 lines per second, so 2,000 lines is the last 80
# SECONDS of an 11-minute condition -- the minutes 1 to 6 where the rejections
# actually happened were never written to disk.
#
# Four changes, all of them instrumentation; the measured condition is identical:
#   a. the training server's log is saved whole, plus a distilled trajectory of
#      one line per retrain (sample count, coverage, violation rate, timestamp);
#   b. that log is also snapshotted after the pre-run, so a kubelet rotation in
#      the measured window cannot take the early part with it;
#   c. the EPP's own view is sampled every 5 s for the whole measured window
#      instead of scraped once at the end. The gauge that matters is
#      inference_objective_inference_request_metric{type="predicted_tpot"}: it
#      reads 47.8 ms at rest while the chat class's per-token budget is 50 ms, so
#      the predictor sits just under the threshold that makes the EPP shed, and a
#      single-point scrape at the end of the run cannot show which side of it the
#      model was on during minutes 1 to 6;
#   d. the prediction servers' logs are saved too -- they serve the model the EPP
#      actually queries, and EXP-108 restarted all three pods together without
#      ever reading two of them.
#
# Usage:
#   ./run_exp110_llmd_lowrate.sh lowrate <n>   n repeats at 10 req/s
#   ./run_exp66_llmd.sh smoke              one condition at 45 req/s
#   ./run_exp66_llmd.sh rep 1              all eight rates, repeat 1
#   ./run_exp66_llmd.sh rep 2              all eight rates, repeat 2
set -uo pipefail

NS=llumnix
LLMD_NS=llmd
HERE="$(cd "$(dirname "$0")" && pwd)"
HOSTWORK=/home/nxclab/llumnix_reproduce/Agent_applications/agent_motivation_experiment
# m1f, not m1. swe's real SLO is E2E <= 30 s and the analysis scores it that way
# whatever this file says, but llm-d cannot express an end-to-end budget: it takes
# x-llm-d-slo-tpot-ms literally. m1's decomposition (11,800, 25) puts the per-token
# target at the idle-state decode ITL median measured in EXP-16, which is exceeded
# almost immediately under load -- the 2026-08-07 smoke ran on m1 and llm-d
# rejected 17-49% of the swe classes for exactly that reason. Llumnix SLO hit the
# same wall first and m1f was written for it: (2,500, 52) is the pair closest to
# FluidServe's nominal pace that still fits 30 s at the measured output length,
# 2,500 + 520.2 x 52 = 29,550. Using it here gives llm-d the same treatment as the
# other baseline that cannot state an end-to-end budget.
# Full account: ms_dev/notes/llmd-baseline.md section 3.3, CLAUDE.md trap group A.
# EXP-108. The per-token form of the agent class's promise: TTFT 7 s + 75 ms per
# token. llm-d reads slo.<class>.tbt_ms literally as the per-token target -- it
# has no way to express an end-to-end budget -- so this file states 75. The
# slofair file it used before states 52, which was the restatement of the old
# 30 s end-to-end budget. Scoring moves with it: FS_SWE_TBT_MS=75.
WCFG=/work/workload_configs/mix_short_m1_t75fair.json
# The hour-trace pair, used only by the `full` entry point below.
DYN_TRACE=/work/traces/dynamic/canonical/dyn60_short_m123_b1045.csv
DYN_WCFG=/work/workload_configs/mix_dyn60_short_m123_b1045_slofair.json
ENVOY_LOG=/home/nxclab/tools/llmd-envoy/envoy_access.log
META="$HOSTWORK/results/exp66_meta"
# The fleet's size comes from configmap/llumnix-model (written by
# ms_dev/scripts/switch_model.py), not from a literal here: with a literal four,
# an eight-instance fleet gets four endpoints registered with llm-d and half the
# fleet is simply never routed to, which looks like a policy result.
ENGINE_PORTS=$(kubectl -n llumnix get cm llumnix-model -o jsonpath='{.data.ENGINE_PORTS}' 2>/dev/null | tr ',' ' ')
[ -n "$ENGINE_PORTS" ] || ENGINE_PORTS="8000 8001 8002 8003"
echo "[driver] engine ports: $ENGINE_PORTS"

# EXP-53's eight rates, in rpm. Kept in this order so a partial run is a prefix
# of the full one and the low rates, which are the cheap ones, land first.
# EXP-108: the eight rates of the fixed static set (10 15 20 25 35 45 55 70
# req/s), NOT the list this driver shipped with. The llm-d column of
# paper_experiment/static_sweep_2026-08 was filled by EXP-68 (35-70) and
# EXP-70/80 (10-25) rather than by this driver's default, so keeping the default
# would produce a curve that cannot be laid over the other arms'.
RATES_RPM="600 900 1200 1500 2100 2700 3300 4200"
PRERUN_MIN=3
MEASURE_MIN=8

mkdir -p "$META"

say() { echo "[exp66] $(date -u +%H:%M:%S) $*"; }

engine_ip() { kubectl -n $NS get pod neutral-0 -o jsonpath='{.status.podIP}' 2>/dev/null; }

# --- preflight ---------------------------------------------------------------
# Refuses to start on the states that have silently produced meaningless runs
# before: a full disk, an engine-side scheduler left on, and -- specific to this
# experiment -- an llm-d stack that is not actually the slo arm.
preflight() {
  kubectl get nodes -o jsonpath='{.items[*].status.conditions[?(@.type=="DiskPressure")].status}' \
    | grep -q True && { say "ABORT: node has DiskPressure"; return 1; }

  local extra
  extra=$(kubectl -n $NS get lws neutral -o json | python3 -c "
import json,sys
c=[c for c in json.load(sys.stdin)['spec']['leaderWorkerTemplate']['workerTemplate']['spec']['containers'] if c['name']=='vllm'][0]
print(next((e.get('value','') for e in c.get('env',[]) if e['name']=='SCHED_EXTRA_ARGS'),'unset'))")
  [ "$extra" = "" ] || { say "ABORT: engine has SCHED_EXTRA_ARGS='$extra', want empty (stock FIFO)"; return 1; }

  local mig
  mig=$(kubectl -n $NS get lws neutral -o json | python3 -c "
import json,sys
c=[c for c in json.load(sys.stdin)['spec']['leaderWorkerTemplate']['workerTemplate']['spec']['containers'] if c['name']=='vllm'][0]
print(next((e.get('value','') for e in c.get('env',[]) if e['name']=='LLUMNIX_ENABLE_MIGRATION'),'unset'))")
  [ "$mig" = "0" ] || { say "ABORT: engine migration is '$mig', want 0"; return 1; }

  # The arm is the ConfigMap the EPP mounts. Reading the label is not enough --
  # the plugin list is what decides behaviour -- so both are checked.
  local arm
  arm=$(kubectl -n $LLMD_NS get cm llmd-epp-config -o jsonpath='{.metadata.labels.llmd\.arm}' 2>/dev/null)
  [ "$arm" = "slo" ] || { say "ABORT: llmd-epp-config is arm '$arm', want 'slo'"; return 1; }
  kubectl -n $LLMD_NS get cm llmd-epp-config -o jsonpath='{.data.config\.yaml}' \
    | grep -q "latency-slo-admitter" \
    || { say "ABORT: EPP config has no latency-slo-admitter"; return 1; }
  kubectl -n $LLMD_NS get cm llmd-epp-config -o jsonpath='{.data.config\.yaml}' \
    | grep -q "streamingMode: true" \
    || { say "ABORT: EPP config does not set streamingMode: true"; return 1; }

  # Path B, i.e. rejection is possible at all.
  kubectl -n $NS get inferencepool llmd-engines >/dev/null 2>&1 \
    || { say "ABORT: InferencePool llmd-engines missing"; return 1; }
  local prio
  prio=$(kubectl -n $NS get inferenceobjective sheddable -o jsonpath='{.spec.priority}' 2>/dev/null)
  [ "$prio" = "-1" ] || { say "ABORT: InferenceObjective sheddable priority is '$prio', want -1"; return 1; }

  [ -s "$ENVOY_LOG" ] || touch "$ENVOY_LOG"
  say "preflight ok: arm=slo, pool present, sheddable priority=-1, engine stock FIFO, migration off"
}

# --- restarts ----------------------------------------------------------------
restart_engine() {
  local old
  old=$(kubectl -n $NS get pod neutral-0 -o jsonpath='{.metadata.uid}' 2>/dev/null)
  kubectl -n $NS delete pod neutral-0 --wait=false >/dev/null 2>&1
  local i uid ready
  for i in $(seq 1 240); do
    uid=$(kubectl -n $NS get pod neutral-0 -o jsonpath='{.metadata.uid}' 2>/dev/null)
    ready=$(kubectl -n $NS get pod neutral-0 -o jsonpath='{.status.containerStatuses[*].ready}' 2>/dev/null)
    [ -n "$uid" ] && [ "$uid" != "$old" ] && [ "$ready" = "true true" ] && break
    sleep 5
  done
  [ "$ready" = "true true" ] || { say "ABORT: neutral-0 not ready after restart"; return 1; }
  # Ready is not serving: wait for all four API servers to answer.
  local ip p ok
  ip=$(engine_ip)
  for i in $(seq 1 240); do
    ok=1
    for p in $ENGINE_PORTS; do
      curl -sf -m 3 -o /dev/null "http://$ip:$p/v1/models" || ok=0
    done
    [ "$ok" = "1" ] && break
    sleep 5
  done
  [ "$ok" = "1" ] || { say "ABORT: engines not serving after restart"; return 1; }
  say "engine restarted and serving on all four ports (ip $ip)"
}

# The three places predictor state lives. Restarting one or two of them leaks:
# the prediction servers keep serving a cached model after the training server
# comes back empty, and the EPP keeps an unflushed sample buffer.
restart_predictor_stack() {
  kubectl -n $LLMD_NS rollout restart deploy/llmd-training-server deploy/llmd-prediction-server deploy/llmd-router >/dev/null
  local d
  for d in llmd-training-server llmd-prediction-server llmd-router; do
    kubectl -n $LLMD_NS rollout status "deploy/$d" --timeout=300s >/dev/null \
      || { say "ABORT: $d did not roll out"; return 1; }
  done
  say "predictor stack restarted (training, prediction x3, epp)"
}

# Direct evidence that the reset happened, rather than the assumption that a
# rollout implies it. The training server logs its sample count once a second;
# the first line after a restart has to say zero.
verify_predictor_empty() {
  local i n
  for i in $(seq 1 30); do
    n=$(kubectl -n $LLMD_NS logs deploy/llmd-training-server --tail=200 2>/dev/null \
        | grep -oE "only [0-9]+ samples" | head -1 | grep -oE "[0-9]+")
    [ -n "$n" ] && break
    sleep 2
  done
  [ -n "$n" ] || { say "ABORT: training server never logged a sample count"; return 1; }
  [ "$n" = "0" ] || { say "ABORT: training server came back with $n samples, want 0"; return 1; }
  say "verified: training server restarted with 0 samples"
}

# --- drain -------------------------------------------------------------------
# A fixed sleep cannot tell whether it was long enough. This reads the engines.
drain() {
  local ip i p run wait total
  ip=$(engine_ip)
  for i in $(seq 1 100); do  # 100 x 3 s = 5 min ceiling
    total=0
    for p in $ENGINE_PORTS; do
      run=$(curl -sf -m 3 "http://$ip:$p/metrics" | awk '/^vllm:num_requests_running/{s+=$2} END{printf "%d", s+0}')
      wait=$(curl -sf -m 3 "http://$ip:$p/metrics" | awk '/^vllm:num_requests_waiting/{s+=$2} END{printf "%d", s+0}')
      total=$(( total + run + wait ))
    done
    [ "$total" = "0" ] && { say "drained after $(( i * 3 ))s"; return 0; }
    sleep 3
  done
  say "WARNING: engines still hold $total requests after 5 min; continuing"
}

# --- job control -------------------------------------------------------------
wait_job() {  # $1 job, $2 deadline minutes
  local job=$1 deadline=$(( $2 * 60 / 10 )) i st
  for i in $(seq 1 "$deadline"); do
    st=$(kubectl -n $NS get job "$job" -o jsonpath='{.status.conditions[*].type}' 2>/dev/null)
    case "$st" in
      *Complete*) return 0 ;;
      *Failed*)   say "$job reached condition Failed"; return 1 ;;
    esac
    sleep 10
  done
  say "$job did not finish within $2 minutes"; return 1
}

launch() {  # $1 job, $2 session, $3 rates rpm, $4 durmin
  local job=$1 session=$2 rates=$3 durmin=$4
  kubectl -n $NS delete job "$job" --ignore-not-found >/dev/null
  sed -e "s/__JOBNAME__/$job/" -e "s/__SESSION__/$session/" \
      -e "s/__RATES__/$rates/" -e "s/__DURMIN__/$durmin/" -e "s/__ARM__/llmdslot75/" \
      -e "s#__WCFG__#$WCFG#" \
      "$HERE/runner-exp66.template.yaml" | kubectl apply -f - >/dev/null
}

# --- predictor sampling ------------------------------------------------------
# EXP-108 scraped the EPP once, after the condition ended, so every question of
# the form "what was the predictor saying while it was rejecting" had a single
# answer covering eight minutes. This samples the same endpoint every 5 s for the
# duration of the measured window and writes one block per sample, each headed by
# a wall-clock epoch so it can be aligned with the client's per-request arrival
# times in metrics.csv.
# Set in condition(); declared here because collect() reads it and the `full`
# entry point reaches collect() without going through condition(), where an
# unset name is a fatal error under set -u.
PRERUN_SNAP=""
SAMPLE_TMP=""
SAMPLER_PID=""
start_sampler() {
  SAMPLE_TMP=$(mktemp -d /home/nxclab/tools/llmd-sample.XXXXXX)
  local eppip; eppip=$(kubectl -n $LLMD_NS get svc llmd-router -o jsonpath='{.spec.clusterIP}')
  (
    while true; do
      printf '# t=%s\n' "$(date -u +%s.%N)"
      curl -sf -m 4 "http://$eppip:9090/metrics" \
        | grep -E '^inference_objective_(inference_request_metric|request_predicted_(ttft|tpot)_seconds_(sum|count)|request_slo_violation_total|request_(ttft|tpot)_seconds_(sum|count)|request_duration_seconds_(sum|count))'
      sleep 5
    done
  ) > "$SAMPLE_TMP/epp_series.txt" 2>/dev/null &
  SAMPLER_PID=$!
  say "EPP sampler started (pid $SAMPLER_PID, every 5 s)"
}
stop_sampler() {
  [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null
  wait "$SAMPLER_PID" 2>/dev/null
  SAMPLER_PID=""
}

# One line per retrain: when, which model, how many samples it was fitted on, and
# how far the fitted quantile is from its 90% target. This is the trajectory the
# missing log was needed for, in a form that can be read without loading 40,000
# lines.
distil_training() {  # $1 source log, $2 destination tsv
  awk -F' - ' '
    /model trained on/ {
      ts=$1
      line=$0
      # Six models are retrained, not two: TTFT and TPOT plus the four gated
      # ensemble sub-models (ttft/tpot x queued/noqueue). Labelling them by the
      # word in front of "model trained on" keeps them apart; collapsing them to
      # ttft/tpot interleaves six series into two and makes the sample count
      # jump back and forth for no reason.
      model = "?"
      if (match(line, /[A-Za-z_]+ model trained on/)) {
        model = substr(line, RSTART, RLENGTH); sub(/ model trained on/, "", model); model = tolower(model)
      }
      n=""; cov=""; vio=""
      if (match(line, /trained on [0-9]+ samples/)) { s=substr(line,RSTART,RLENGTH); gsub(/[^0-9]/,"",s); n=s }
      if (match(line, /Coverage = [0-9.]+%/))       { s=substr(line,RSTART,RLENGTH); gsub(/[^0-9.]/,"",s); cov=s }
      if (match(line, /Violation Rate = [0-9.]+%/)) { s=substr(line,RSTART,RLENGTH); gsub(/[^0-9.]/,"",s); vio=s }
      print ts "\t" model "\t" n "\t" cov "\t" vio
    }' "$1" > "$2"
}

# --- collection --------------------------------------------------------------
# Envoy's file access logger buffers for about ten seconds, so copying the
# moment a job ends drops the tail of the condition -- and the loss looks like a
# routing failure rather than a copy that was too early. Wait for the size to
# stop moving, then take the slice this condition wrote.
collect() {  # $1 session, $2 envoy byte offset at condition start
  local session=$1 off=$2
  local dir a b i
  dir=$(ls -dt "$HOSTWORK"/results/*"${session}"* 2>/dev/null | head -1)
  [ -n "$dir" ] || { say "WARNING: no result dir for $session"; return 1; }

  a=$(stat -c%s "$ENVOY_LOG"); sleep 12
  for i in $(seq 1 10); do
    b=$(stat -c%s "$ENVOY_LOG")
    [ "$a" = "$b" ] && break
    a=$b; sleep 6
  done
  tail -c "+$(( off + 1 ))" "$ENVOY_LOG" > "$dir/envoy_access.log"
  local n; n=$(wc -l < "$dir/envoy_access.log")

  local eppip; eppip=$(kubectl -n $LLMD_NS get svc llmd-router -o jsonpath='{.spec.clusterIP}')
  curl -sf -m 10 "http://$eppip:9090/metrics" > "$dir/epp_metrics.txt" || say "WARNING: EPP metrics scrape failed"
  kubectl -n $LLMD_NS get cm llmd-epp-config -o jsonpath='{.data.config\.yaml}' > "$dir/epp_config.yaml"
  # WHOLE log, not --tail. See the header: this log runs at about 25 lines per
  # second, so the 2,000 lines EXP-108 kept were its last 80 seconds.
  kubectl -n $LLMD_NS logs deploy/llmd-training-server --timestamps > "$dir/training_server.log" 2>/dev/null
  distil_training "$dir/training_server.log" "$dir/training_trajectory.tsv"
  kubectl -n $LLMD_NS logs deploy/llmd-prediction-server --timestamps --all-containers \
    > "$dir/prediction_server.log" 2>/dev/null
  kubectl -n $LLMD_NS logs deploy/llmd-router --timestamps > "$dir/epp.log" 2>/dev/null
  [ -f "$PRERUN_SNAP" ] && mv "$PRERUN_SNAP" "$dir/training_server_after_prerun.log"
  if [ -n "$SAMPLE_TMP" ] && [ -f "$SAMPLE_TMP/epp_series.txt" ]; then
    mv "$SAMPLE_TMP/epp_series.txt" "$dir/epp_series.txt"
    rmdir "$SAMPLE_TMP" 2>/dev/null
  fi

  local tl es
  tl=$(wc -l < "$dir/training_server.log" 2>/dev/null || echo 0)
  es=$(grep -c '^# t=' "$dir/epp_series.txt" 2>/dev/null | head -1)
  say "collected into $(basename "$dir"): envoy lines=$n training lines=$tl retrains=$(wc -l < "$dir/training_trajectory.tsv") epp samples=${es:-0}"
}

# --- one condition -----------------------------------------------------------
condition() {  # $1 rate rpm, $2 session
  local rate=$1 session=$2
  say "=== condition $session rate=${rate}rpm"
  restart_engine            || return 1
  restart_predictor_stack   || return 1
  verify_predictor_empty    || return 1

  say "pre-run ${PRERUN_MIN}min at ${rate}rpm (discarded)"
  launch "bench-runner-exp108-prerun" "${session}_PRERUN" "$rate" "$PRERUN_MIN"
  wait_job "bench-runner-exp108-prerun" 30 || { say "pre-run failed"; return 1; }
  kubectl -n $NS delete job bench-runner-exp108-prerun --ignore-not-found >/dev/null

  local trained
  trained=$(kubectl -n $LLMD_NS logs deploy/llmd-training-server --tail=50 2>/dev/null \
            | grep -oE "only [0-9]+ samples" | tail -1 | grep -oE "[0-9]+")
  say "after pre-run: training server sample line = '${trained:-<training, not skipping>}'"

  # Insurance against a kubelet log rotation during the measured window taking
  # the pre-run and the first minutes with it. The predictor pod is restarted per
  # condition, so its log is short, but the whole point of this experiment is the
  # early part of it and a second copy costs a few megabytes.
  PRERUN_SNAP=/home/nxclab/tools/llmd-prerun-$$.log
  kubectl -n $LLMD_NS logs deploy/llmd-training-server --timestamps > "$PRERUN_SNAP" 2>/dev/null

  drain
  local off; off=$(stat -c%s "$ENVOY_LOG")

  say "measure ${MEASURE_MIN}min at ${rate}rpm"
  start_sampler
  launch "bench-runner-exp108" "$session" "$rate" "$MEASURE_MIN"
  wait_job "bench-runner-exp108" 60 || {
    stop_sampler
    say "measurement failed"; kubectl -n $NS logs job/bench-runner-exp108 --tail=60; return 1; }
  stop_sampler
  kubectl -n $NS logs job/bench-runner-exp108 --tail=400 2>/dev/null | grep -aE "Success:|rc=" | tail -3
  kubectl -n $NS delete job bench-runner-exp108 --ignore-not-found >/dev/null

  collect "$session" "$off"
  say "=== condition $session DONE"
}

# --- entry points ------------------------------------------------------------
case "${1:-}" in
  smoke)
    preflight || exit 1
    condition 2700 "exp66smoke_llmdslo_m1f" || exit 1
    say "SMOKE DONE"
    ;;
  # EXP-68 adds a single-rate entry so a chain can interleave this arm with the
  # FluidServe driver at one rate instead of running the whole sweep. Everything
  # else -- engine restart, predictor stack reset, the 3 min pre-run, the drain,
  # the 8 min measure -- is unchanged, so an llm-d condition here is the same
  # condition EXP-66 ran.
  one)
    preflight || exit 1
    condition "${2:?usage: one <rate-rpm> <session>}" "${3:?session}" || exit 1
    say "ONE DONE"
    ;;
  # EXP-71. The hour-long trace. llm-d had no trace path at all: WCFG above is a
  # static config and `condition` only knows a rate in rpm. This entry swaps both
  # for the dynamic pair -- the m1f workload config, because llm-d cannot express
  # an end-to-end budget and would otherwise read swe's 25 ms tier key as a
  # per-token target, and the trace whose class plan that config names.
  #
  # No pre-run here. The 3 min pre-run exists to give the predictor a warm start
  # at a rate the measured run will then hold; a trace whose rate moves between
  # 10.8 and 45.0 req/s has no such rate, and the trace carries its own 60 s
  # warmup segment. The predictor therefore learns during the measured hour,
  # which is what a deployment would do, and it is stated in the record rather
  # than hidden.
  full)
    session="${2:?usage: full <session>}"
    preflight || exit 1
    say "=== hour trace $session"
    restart_engine            || exit 1
    restart_predictor_stack   || exit 1
    verify_predictor_empty    || exit 1
    drain
    off=$(stat -c%s "$ENVOY_LOG")
    kubectl -n $NS delete job bench-runner-exp108hour --ignore-not-found >/dev/null
    sed -e "s/__JOBNAME__/bench-runner-exp108hour/" -e "s/__SESSION__/$session/" \
        -e "s/__ARM__/llmdslot75/" \
        -e "s#__TRACE__#$DYN_TRACE#" -e "s#__WCFG__#$DYN_WCFG#" \
        "$HERE/runner-exp71-llmd-dyn.template.yaml" | kubectl apply -f - >/dev/null
    wait_job "bench-runner-exp108hour" 120 || {
      say "hour trace failed"; kubectl -n $NS logs job/bench-runner-exp108hour --tail=60; exit 1; }
    kubectl -n $NS logs job/bench-runner-exp108hour --tail=400 2>/dev/null | grep -aE "Success:|rc=" | tail -3
    kubectl -n $NS delete job bench-runner-exp108hour --ignore-not-found >/dev/null
    collect "$session" "$off"
    say "=== hour trace $session DONE"
    ;;
  # EXP-110. The one rate whose repeats disagreed, run enough times that the
  # split can be described rather than guessed at. Each repeat gets its own
  # session name so all of them survive as separate result directories; a repeat
  # that fails does not stop the ones after it, because the question is how often
  # each outcome occurs and a truncated series answers it wrongly.
  lowrate)
    n="${2:?usage: lowrate <repeats>}"
    preflight || exit 1
    ok=0
    for i in $(seq 1 "$n"); do
      if condition 600 "exp110r${i}_llmdslot75_t75fair"; then
        ok=$(( ok + 1 ))
      else
        say "repeat $i failed; continuing to the next"
      fi
    done
    say "=== EXP-110 LOWRATE DONE  ${ok}/${n} conditions produced a result  $(date -u +%Y-%m-%d\ %H:%M:%S)Z"
    say "=== result directories: $(ls -d "$HOSTWORK"/results/*exp110r*_llmdslot75_t75fair* 2>/dev/null | grep -vc PRERUN)"
    ;;
  rep)
    rep="${2:?usage: rep <n>}"
    preflight || exit 1
    for r in $RATES_RPM; do
      condition "$r" "${SESSION_PREFIX_BASE:-exp108}r${rep}_llmdslot75_t75fair" || { say "stopping at ${r}rpm"; exit 1; }
    done
    say "=== EXP-66 REPEAT ${rep} DONE $(date -u +%Y-%m-%d\ %H:%M:%S)Z"
    ;;
  *)
    echo "usage: $0 {smoke|one <rpm> <session>|full <session>|rep <n>}"; exit 2 ;;
esac
