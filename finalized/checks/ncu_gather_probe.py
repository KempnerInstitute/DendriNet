"""Minimal single-shot gather launch, for Nsight Compute to attach to.

roofline_gather.py computes achieved bandwidth from an ANALYTIC traffic model,
and the answer swings from 1.7% to 53% of peak depending on whether you assume
the x-gathers are cached, re-read per block, or not cached at all. That spread
is too wide to conclude anything from, so the model has to be replaced by the
hardware's own counters.

This runs exactly one forward and one backward at the dominant stress-config
shape, with everything before it excluded via cudaProfilerStart/Stop.

    ncu --profile-from-start off \\
        --metrics dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,\\
sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,\\
sm__warps_active.avg.pct_of_peak_sustained_active \\
        python ncu_gather_probe.py
"""

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    cuda_gather as cg,
)

IN, OUT, K, M = 16000, 144000, 40, 512

ext = cg._load_ext()
dev = torch.device("cuda")
idx = torch.randint(0, IN, (OUT, K), device=dev, dtype=torch.int32).contiguous()
wf = (torch.randn(OUT, K, device=dev) * 0.05).contiguous()
xf = torch.randn(M, IN, device=dev).abs().contiguous()
gof = torch.randn(M, OUT, device=dev).contiguous()
goT = gof.t().contiguous()
xT = xf.t().contiguous()
P, T = cg._pick_wpo_params()
Pb, Tb = cg._pick_bwd_wpo_params(OUT)

for _ in range(2):  # warm up outside the profiled region
    ext.gather_fwd_wpo_cuda(xf, idx, wf, P, T)
    ext.gather_bwd_wpo_cuda(gof, goT, xT, idx, wf, Pb, Tb)
torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStart()
ext.gather_fwd_wpo_cuda(xf, idx, wf, P, T)
ext.gather_bwd_wpo_cuda(gof, goT, xT, idx, wf, Pb, Tb)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
