# DEEP NSight Compute capture of the CUDA gather kernels -- the follow-up to ncu_quick.sh.
#
# WHY: the `--set basic` run (ncu_quick.sh) showed both kernels are L2-throughput-bound
#   (L2 ~93-95%, DRAM ~0.8%, compute 6.5-17%). `basic` did NOT capture the Memory Workload
#   Analysis section, which is exactly where the L2 story lives. ncu itself said:
#   "Start by analyzing L2 in the Memory Workload Analysis section." This script captures it.
#
# WHAT changes vs ncu_quick.sh:
#   --set full  (was --set basic) -> adds MemoryWorkloadAnalysis, SchedulerStats,
#               WarpStateStats, SourceCounters, etc. Heavier (more replay passes) but the
#               2 profiled launches make it affordable.
#
# READ AFTER, in this order (offline, no GPU needed):
#   ncu -i cuda_gather_ncu_full.ncu-rep --page details --section MemoryWorkloadAnalysis
#     -> L2 hit rate, sectors/request (coalescing), which access dominates L2 traffic.
#   Key numbers to grab:
#     * L2 Hit Rate                       (high = data resident; low = we're thrashing L2)
#     * L1/TEX Hit Rate                   (low L1 hit = scattered x-gather missing L1 -> L2)
#     * Sectors/Req for global loads      (>4 on a 32-thread warp load = uncoalesced waste)
#     * grad_x atomic traffic (bwd)       (the scatter is the suspected bwd L2 hog)
#
# IMPORTANT (same as ncu_quick.sh):
#   * RUN ON THE FULL BLACKWELL, not the A100 MIG slice (MIG blocks ncu counters).
#   * `source setup_cuda_env.sh` first.
#   * ERR_NVGPUCTRPERM -> perf counters admin-restricted on this node; try another partition.
#
# Usage:  bash ncu_full.sh
set -u
DIR="/n/home12/rwhalen/DendriNet/profilings/weight_transform_opt"
OUT="$DIR/cuda_gather_ncu_full"     # -> cuda_gather_ncu_full.ncu-rep

if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu not found. Run: source setup_cuda_env.sh  (ncu ships with cuda/12.9.1)"; exit 1
fi
echo "ncu     : $(command -v ncu)"; ncu --version | head -1
echo "gpu     : $(python -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null)"
echo

NCU_ITERS=4 ncu --set full \
    -k "regex:gather_(fwd|bwd)_v2_kernel" \
    --launch-skip 4 --launch-count 2 \
    -f -o "$OUT" \
    python "$DIR/ncu_driver.py"
rc=$?

echo
if [ $rc -eq 0 ]; then
    echo "OK -> wrote $OUT.ncu-rep"
    echo
    echo "Now read the L2 story (the thing basic couldn't show):"
    echo "  ncu -i $OUT.ncu-rep --page details --section MemoryWorkloadAnalysis | less"
    echo
    echo "Or pull the key coalescing/hit-rate metrics straight out:"
    echo "  ncu -i $OUT.ncu-rep --page raw --metrics \\"
    echo "    lts__t_sector_hit_rate.pct,\\"
    echo "    l1tex__t_sector_hit_rate.pct,\\"
    echo "    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\\"
    echo "    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio"
else
    echo "ncu exited $rc. If ERR_NVGPUCTRPERM: perf counters admin-restricted here."
    echo "If on the A100 MIG slice: rerun on the full Blackwell GPU."
fi
