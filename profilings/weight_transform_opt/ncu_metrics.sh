# FAST one-pass NSight Compute look at the CUDA gather kernels -- the fast-iteration companion
# to ncu_full.sh. Prints the handful of L2-bound metrics that matter straight to the terminal.
#
# WHY a separate script: `--set full` replays each launch many times (one per metric group) and
# is slow. When you're tweaking the kernel and want a 5-second "did L2 traffic drop?" answer,
# use THIS -- it gathers a hand-picked metric list in as few passes as possible and prints a
# table (no .ncu-rep, no UI round-trip).
#
# The metrics, and what each tells us for our L2-throughput-bound kernels:
#   gpu__time_duration.sum                          -> wall time (did the change actually help?)
#   sm__throughput...pct                            -> Compute SOL (should stay low; we're not compute-bound)
#   gpu__dram_throughput...pct                      -> DRAM SOL (was 0.78% -- expect it to stay ~idle)
#   lts__throughput...pct                           -> L2 SOL (THE wall, was 93/96% -- want this DOWN)
#   l1tex__throughput...pct                         -> L1 SOL (was 52/57%)
#   lts__t_sector_hit_rate.pct                      -> L2 hit rate
#   l1tex__t_sector_hit_rate.pct                    -> L1 hit rate (low = scattered x-gather misses L1 -> L2)
#   ...sectors_per_request...global_op_ld.ratio     -> load coalescing (>4 on the x-gather = wasted L2 traffic)
#   ...sectors_per_request...global_op_st.ratio     -> store coalescing (the out write)
#   ...sectors_per_request...global_op_atom.ratio   -> atomic coalescing (bwd grad_x scatter -- suspected hog)
#
# RUN ON THE FULL BLACKWELL, not the A100 MIG slice. `source setup_cuda_env.sh` first.
#
# Usage:  bash ncu_metrics.sh
set -u
DIR="/n/home12/rwhalen/DendriNet/profilings/weight_transform_opt"

if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu not found. Run: source setup_cuda_env.sh  (ncu ships with cuda/12.9.1)"; exit 1
fi
echo "ncu     : $(command -v ncu)"; ncu --version | head -1
echo "gpu     : $(python -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null)"
echo

METRICS="gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,\
l1tex__t_sector_hit_rate.pct,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_atom.ratio"

# --metrics : single hand-picked list, far fewer passes than --set full.
# -k regex  : both v2 kernels.  -s 4 -c 2 : skip warm-up, profile next fwd+bwd (warm caches).
NCU_ITERS=4 ncu --metrics "$METRICS" \
    -k "regex:gather_(fwd|bwd)_v2_kernel" \
    --launch-skip 4 --launch-count 2 \
    python "$DIR/ncu_driver.py"
rc=$?

echo
if [ $rc -ne 0 ]; then
    echo "ncu exited $rc. If ERR_NVGPUCTRPERM: perf counters admin-restricted here."
    echo "If a metric name errors: ncu --query-metrics | grep <partial-name>  to find the exact id."
    echo "If on the A100 MIG slice: rerun on the full Blackwell GPU."
fi
