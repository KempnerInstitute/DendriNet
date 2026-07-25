# Quick NSight Compute look at the production CUDA gather kernels -- to scope tomorrow's
# deep-dive. Prints a SpeedOfLight/occupancy/memory summary to the terminal AND writes a
# .ncu-rep report to open in the NSight Compute UI.
#
# IMPORTANT:
#   * RUN ON THE BLACKWELL (full GPU), NOT the A100 MIG slice. ncu's hardware counters are
#     restricted/unavailable under MIG -- it will likely error there.
#   * `source setup_cuda_env.sh` first (ncu ships with the cuda module).
#   * If ncu errors with ERR_NVGPUCTRPERM, GPU perf-counter access is admin-restricted on
#     this node -- try another GPU partition, or ask FASRC to enable non-admin profiling.
#
# Usage:  bash ncu_quick.sh
set -u
DIR="/n/home12/rwhalen/DendriNet/profilings/weight_transform_opt"
OUT="$DIR/cuda_gather_ncu"     # -> cuda_gather_ncu.ncu-rep

if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu not found. Run: source setup_cuda_env.sh  (ncu ships with cuda/12.9.1)"; exit 1
fi
echo "ncu     : $(command -v ncu)"; ncu --version | head -1
echo "gpu     : $(python -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null)"
echo

# --set basic : SpeedOfLight (compute vs memory bound), occupancy, memory workload -- fast, high-signal.
# -k regex    : both v2 kernels.
# -s 4 -c 2   : skip 2 warm-up iters (4 matched launches), profile the next fwd+bwd (warm caches).
# -f -o       : overwrite + write the report file.
NCU_ITERS=4 ncu --set basic \
    -k "regex:gather_(fwd|bwd)_v2_kernel" \
    --launch-skip 4 --launch-count 2 \
    -f -o "$OUT" \
    python "$DIR/ncu_driver.py"
rc=$?

echo
if [ $rc -eq 0 ]; then
    echo "OK -> wrote $OUT.ncu-rep"
    echo "  open in NSight Compute UI, or re-print the summary with:"
    echo "    ncu -i $OUT.ncu-rep --page details | less"
    echo
    echo "For tomorrow, the first things to read per kernel:"
    echo "  * SpeedOfLight  : is it Compute- or Memory-bound? (tells us which wall we're on)"
    echo "  * Memory        : DRAM throughput % of peak, and % of sectors from uncoalesced access"
    echo "  * Occupancy     : achieved vs theoretical (shared-mem/register limited?)"
else
    echo "ncu exited $rc. If ERR_NVGPUCTRPERM: perf counters are admin-restricted here."
    echo "If on the A100 MIG slice: rerun on the full Blackwell GPU (MIG blocks ncu counters)."
fi
