#!/bin/bash
# Validate the memory work against the REAL trainer, not the profiling harness.
#
# Every number in the reactivation/shunt work came from a synthetic step:
#   out.float().pow(2).mean().backward()
# with no optimizer, no real loss, no data loader. That measures the
# forward/backward core, which is what the work targeted -- but it is not a
# training step. This runs train_experiments.py for one real epoch (Adam,
# cross-entropy, real CIFAR-10 batches) twice, and compares:
#
#   * peak GPU memory   -- does the 2.19x survive Adam's state and the loss head?
#   * epoch wall-clock  -- does the speed neutrality survive real data loading?
#   * the LOSS TRAJECTORY -- the strongest equivalence evidence available. The
#     fused paths differ from eager by ~1e-7 rounding; over a real epoch of Adam
#     updates that either stays invisible or it does not, and only this shows it.
#
# The two runs differ ONLY by DENDRINET_POSITIVE_TANH_SIGMOID, which gates all
# three fused paths, so no code or config changes between them.
#
#   bash real_train_ab.sh <outdir>
set -uo pipefail
OUT=${1:-profilings/dendrinet_harness/ff_stress/realtrain}
CFG=configs/examples/kernel_stress_ab/kernel_stress_12L_16k_cuda_1ep.yaml
PY=/n/home12/rwhalen/.conda/envs/dendritic_env/bin/python
mkdir -p "$OUT"
export PYTHONPATH="src:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

run() {  # run <tag> <sigmoid-flag>
  # Separate declarations: chained `local a=$1 b="$a"` does not reliably see the
  # earlier name under `set -u`.
  local tag="$1"
  local flag="$2"
  local log="$OUT/$tag.log"
  local pf
  if [ -s "$log" ] && grep -q "REAL_TRAIN_DONE" "$log"; then
    echo "===== SKIP $tag (complete) ====="; return
  fi
  echo "===== RUN $tag (DENDRINET_POSITIVE_TANH_SIGMOID=$flag) ====="
  pf=$(mktemp)
  # Poll for peak from OUTSIDE the process: the trainer does not report it, and
  # torch's own counter is not reachable without editing the trainer.
  ( m=0; while :; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      case "$u" in ""|*[!0-9]*) ;; *) [ "$u" -gt "$m" ] && m=$u; echo "$m" > "$pf";; esac
      sleep 0.5
    done ) & poller=$!

  local t0=$SECONDS
  # Run through the wrapper so torch's own max_memory_allocated is reported.
  # nvidia-smi sees the allocator's RESERVED pool, which is dominated by caching
  # and is not the quantity every other figure in this work reports.
  DENDRINET_POSITIVE_TANH_SIGMOID=$flag $PY -u \
    profilings/dendrinet_harness/rnn_prof/run_trainer_measured.py "$CFG" \
    --output_dir "$OUT/$tag" --run_name "$tag" > "$log" 2>&1
  local rc=$?
  local elapsed=$((SECONDS - t0))
  kill $poller 2>/dev/null; wait $poller 2>/dev/null

  local torchpeak
  torchpeak=$(grep -oE "TORCH_PEAK_ALLOCATED_MB=[0-9.]+" "$log" | tail -1 | cut -d= -f2)
  echo "REAL_TRAIN_DONE tag=$tag exit=$rc elapsed_s=$elapsed smi_peak_mb=$(cat "$pf" 2>/dev/null) torch_peak_mb=${torchpeak:-NA}" \
    | tee -a "$log"
  rm -f "$pf"
  [ $rc -ne 0 ] && { echo "  --- tail ---"; tail -15 "$log"; }
}

run eager 0
run eager_control 0   # same code, different run -- establishes the noise floor
run fused 1

echo
echo "================ SUMMARY ================"
for t in eager eager_control fused; do
  grep -h "REAL_TRAIN_DONE" "$OUT/$t.log" 2>/dev/null
done
