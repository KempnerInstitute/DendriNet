#!/bin/bash
# Close the three declared picker edges on whatever GPU this lands on.
#
# The forward-gather picker (_pick_fwd_variant) is a lookup table built ONLY
# from measurement. Three cells were declared unmeasured when the CUDA track
# closed:
#   (a) H100/H200: no data -> picker falls back to staged
#   (b) batch: every band was swept at batch 512 only
#   (c) K band edges: 20 and 60 were interpolated from K in {10, 40, 80}
# One run of this script produces every table needed to fill those cells for
# the architecture it lands on. Run it once per arch that matters: an H100 run
# closes (a) outright (this sweep covers batch AND K, so the new arch gets no
# interpolated cells); a Blackwell re-run stamps (b)+(c) where the current
# bands were fitted.
#
#   bash sweep_picker_edges.sh <outdir>
#
# Resume-safe: completed phase logs are skipped, so a preempted gpu_requeue
# job just reruns. ~45 min on one GPU (+ ~3 min one-time JIT build on a new
# arch).
set -uo pipefail
OUT=${1:?outdir}
mkdir -p "$OUT"
cd "$(dirname "$0")/../../.."   # repo root
source finalized/setup_cuda_env.sh
export PYTHONPATH="src:${PYTHONPATH:-}"
HERE=profilings/dendrinet_harness/rnn_prof
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tee "$OUT/gpu.txt"

step() { # step <logname> <cmd...>
  local log="$OUT/$1.log"; shift
  if [ -s "$log" ]; then echo "SKIP $log"; return; fi
  echo "== $log =="
  if "$@" > "$log" 2>&1; then
    tail -4 "$log"
  else
    echo "  FAILED:"; tail -8 "$log"; mv "$log" "$log.failed"
  fi
}

# [A] arch table -- full grid at batch 512, the exact sweep Blackwell/A100 got
for K in 10 40 80; do
  step "A_crossover_b512_K$K" python "$HERE/bench_gather_crossover.py" --batch 512 --K "$K"
done

# [B] batch dependence -- K=40 full grid at the batches the bands never saw
for B in 64 128; do
  step "B_crossover_b${B}_K40" python "$HERE/bench_gather_crossover.py" --batch "$B" --K 40
done

# [C] K band edges -- dense K on the 1-blk/SM shapes where the bands decide
for K in 15 20 25 30 50 60 70; do
  step "C_edges_b512_K$K" python "$HERE/bench_gather_crossover.py" --batch 512 --K "$K" \
    --ins 16000,24000 --outs 2000,48000,144000
done

# [D] hybrid -- does the prefix-staged kernel earn a band on this arch/batch?
step "D_hybrid_b512" python "$HERE/bench_gather_hybrid.py" --batch 512
step "D_hybrid_b128" python "$HERE/bench_gather_hybrid.py" --batch 128

echo
echo "all logs in $OUT -- feed the directory back for picker-band fitting"
