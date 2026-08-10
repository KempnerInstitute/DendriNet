#!/bin/bash
# Four-arm figure run: loop (baseline) vs fused vs triton vs cuda on ONE
# feedforward config. Same config file, same seeds, same iteration counts --
# the ONLY thing that differs between arms is the gather path (--path
# override on the flag-neutral _off.yaml). Produces the speedup and
# peak-memory figures vs loop, plus the equivalence receipt that justifies
# the claim "only the gather level changed" (outputs bit-identical /
# fp-reassociation bounded across paths).
#
#   bash finalized/harness/figure_three_levels.sh <outdir>
#
# ~30-60 min on one GPU (the loop baseline at batch 512 dominates the time).
set -uo pipefail
OUT=${1:?outdir}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."   # repo root
source finalized/setup_cuda_env.sh
export PYTHONPATH="src:${PYTHONPATH:-}"
CFG=finalized/configs/kernel_stress_12L_16k_off.yaml
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CAP_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU: $GPU (${CAP_MB} MB)" | tee "$OUT/gpu.txt"

step() { # step <logname> <cmd...>
  local log="$OUT/$1.log"; shift
  if [ -s "$log" ] && ! grep -q "Traceback" "$log"; then echo "SKIP $log"; return; fi
  echo "== $log =="
  if "$@" > "$log" 2>&1; then tail -3 "$log"; else echo "  FAILED:"; tail -8 "$log"; mv "$log" "$log.failed"; fi
}

# [0] the receipt: gather paths are equivalent, so arms differ ONLY in speed/mem
step equivalence python profilings/dendrinet_harness/rnn_prof/check_module_equiv.py

# [1] four arms, identical invocation except --path
for path in loop fused triton cuda; do
  step "profile_$path" python finalized/harness/profile_dendrinet.py \
    --configs "$CFG" --path "$path" --batches 128 512 --outdir "$OUT"
done

# [2] figures (speedup bars + peak-memory bars + the rest of the standard set)
step figures python finalized/harness/make_ff_figures.py \
  --root "$OUT" --gpu "$GPU" --capacity-mb "$CAP_MB" --outdir "$OUT/figures"

echo
echo "figures in $OUT/figures ; equivalence receipt in $OUT/equivalence.log"
