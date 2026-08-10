# rnn_prof/ — misnamed; this is the kernel/optimization lab bench

Named when RNN profiling was the active front (July 2026); it has since
accumulated ALL kernel and optimization work — feedforward included: the CUDA
gather crossover/hybrid/picker-edge sweeps (`bench_gather_*`, `logs/edges_*`),
the closeout overlay, and the Triton v3 benches (`bench_triton_*`). Kept in
place because the ledger (`../rnn_scaling_benchmark.md`) and memory cite these
paths as evidence.

This is scratch space: experiments, logs, and dead ends. The curated,
lab-facing copies of everything that matters live in `finalized/` at the repo
root — start there.
