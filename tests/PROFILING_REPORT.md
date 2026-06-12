# Profiling DendriNet — An Explainer

> **How to use this document.** This is written so you can paste it into a
> Claude chat (claude.ai) and have a detailed conversation about it, even though
> that Claude can't see the codebase. It explains *what* we found, *why* it
> matters, and *what* the profiling plan is — in plain language, defining the
> jargon as it goes. If you want to ask Claude "why does X matter?" or "explain
> Y more", everything it needs is in here.
>
> Context: **DendriNet** (package name `dendritic_modeling`) is a PyTorch
> research codebase for biologically-inspired neural networks (dendritic
> excitation–inhibition models). The goal is **not** to change the model — it's
> to *profile* training to understand where the time goes, as a learning
> exercise. We investigated before instrumenting anything.

---

## 1. What "profiling" means here, and the plan in one picture

Profiling = measuring where a program actually spends its time, instead of
guessing. For deep-learning training, the central question is usually:

> **Is the GPU actually busy doing math, or is it sitting idle waiting for the
> CPU / data / Python?**

A GPU that's idle 60% of the time is the single most common (and most fixable)
performance problem in ML training, and you can't see it without measuring.

We're using a **"profiling ladder"** — three tools, increasing in detail and
cost, and we only climb as far as the workload justifies:

| Rung | Tool | What it shows | Analogy |
|---|---|---|---|
| **1** | `torch.profiler` | Per-operation CPU vs GPU time; whether the GPU idles | A labeled receipt: which operations cost what |
| **2** | **Nsight Systems** (`nsys`) | A full timeline of CPU threads, GPU streams, data copies, and the *gaps* between them | A flight tracker: see everything moving (or stalled) over time |
| **3** | **Nsight Compute** (`ncu`) | The internals of *one specific GPU kernel* (memory vs compute bound, occupancy) | An MRI of a single organ |

The key insight driving the plan: **Rung 3 is only worth it if you have a custom
GPU kernel you can actually rewrite.** Profiling a standard, vendor-provided
kernel (like NVIDIA's matrix-multiply) at that depth is pointless — you can't
change it. So a big part of the investigation was determining whether this
codebase has any custom kernels at all. (Spoiler: it doesn't, so we expect to
stop at Rung 2.)

---

## 2. Key terms (so the rest reads smoothly)

- **PyTorch eager mode** — the default way PyTorch runs: each operation executes
  immediately, one at a time, dispatched from Python. Flexible, but every op
  pays a small "launch overhead" cost. Lots of tiny ops → lots of overhead.
- **Kernel** — a single function that runs on the GPU (e.g. one matrix multiply,
  one element-wise add). Training a network = launching thousands of kernels.
- **Kernel-launch overhead** — the fixed CPU cost of telling the GPU to start a
  kernel. If your kernels are *small*, this overhead can dominate, and the GPU
  spends more time waiting to be told what to do than actually computing.
- **`.item()` / GPU→CPU sync** — calling `.item()` on a GPU tensor forces the CPU
  to *stop and wait* until the GPU finishes everything so far, then copies one
  number back. It's a "synchronization point." Done every step, it can serialize
  the CPU and GPU that would otherwise overlap. (We found this — see §4.)
- **AMP (Automatic Mixed Precision)** — running parts of the model in 16-bit
  instead of 32-bit floats for speed/memory, with safeguards. Optional here.
- **DataLoader / `num_workers`** — PyTorch's data-feeding machinery. Worker
  processes prepare the next batch on the CPU while the GPU trains on the
  current one. Too few workers → GPU starves waiting for data.
- **Custom kernel** — a GPU function *you* wrote (in CUDA C++ or Triton), as
  opposed to the standard ones PyTorch ships. Only these are worth Rung 3.
- **NVTX ranges** — labeled markers you insert in code ("forward starts here",
  "backward ends here") so the Nsight timeline is human-readable instead of an
  undifferentiated wall of kernels.

---

## 3. What this codebase is (the investigation, Phase 1)

Before touching anything, we read the repo to answer six questions. Here's what
we found and, importantly, *why each answer shapes the plan*.

### Framework: PyTorch, eager mode
It's PyTorch (version 2.x), using GPUs in the ordinary way — move data and model
to the GPU, run forward/backward. TensorFlow and HuggingFace Transformers appear
only as *optional* add-ons that the training path doesn't use. There's no JAX.
**Why it matters:** all the standard PyTorch profiling tools apply directly;
nothing exotic.

One nuance: the defaults are tuned for **reproducibility, not speed**
(deterministic algorithms on, certain fast-math GPU features off). So some of
the "slowness" we may see is *intentional* — a research correctness choice, not a
bug. Worth keeping in mind so we don't "optimize away" determinism by accident.

### The training loop — where the time is spent
The actual per-batch work (the thing we'll profile) lives in one function:
`Trainer._epoch_train`. Its inner loop, for each batch, does the textbook five
steps: copy data to GPU → forward pass (compute predictions + loss) → zero old
gradients → backward pass (compute new gradients) → optimizer step (update
weights). There are ten specialized trainer variants, but they all build on this
one, so it's the right single place to instrument.

### Data loading — probably *not* the bottleneck
The datasets (MNIST, CIFAR, etc.) are loaded **entirely into memory up front** as
tensors. There's **no per-item disk reading** during training. **Why it matters:**
"my training is slow because of disk I/O" — a very common culprit — is ruled out
here. The data is already in RAM. (There's one optional dataset that does heavier
CPU work per sample, but the config we chose doesn't use it.)

### Custom kernels — the decisive question for Rung 3
We searched hard for anything that would justify the deepest profiling tool:
- **No** hand-written CUDA C++ kernels.
- **No** Triton kernels.
- **One** "custom autograd Function" (a hand-defined gradient, for a spiking
  neuron model) — but it's just ordinary element-wise math, not a real kernel.
- An **optional** use of `torch.compile` (PyTorch's just-in-time optimizer),
  which produces machine-generated code that Nsight Compute can't meaningfully
  inspect anyway.
- The interesting-sounding custom layers (block-sparse linear, top-K linear,
  dendritic convolutions) are all **built by composing standard PyTorch ops** —
  so under the hood they call NVIDIA's stock matrix-multiply and convolution
  kernels, which we can't rewrite.

**Conclusion:** there is no custom kernel worth Nsight Compute. We plan to **skip
Rung 3**. This is exactly the kind of decision that saves you from spending an
afternoon collecting data you can't act on.

### Mixed precision, gradient accumulation, distributed training
- **AMP** (16-bit speedups) exists but is **off by default** — opt-in per config.
- **Gradient accumulation** (simulating big batches on small GPUs) is **not
  implemented** — every batch updates the weights.
- **Distributed training** (DDP / FSDP, i.e. splitting work across multiple GPUs)
  is supported and selected by a config switch, launched via `torchrun`.
- **Why it matters:** for a first, clean profile we deliberately pick a
  *single-GPU, AMP-off* run, so the timeline is simple and interpretable before
  we add complications.

### The one thing that jumped out
In the training loop, after every batch the code calls `.item()` **three times**
(to read the loss values for logging). As explained in §2, each `.item()` is a
GPU→CPU synchronization — it makes the CPU wait for the GPU. Doing this every
single step is a classic, easy-to-miss cause of the GPU and CPU failing to
overlap their work. **We are not "fixing" it** (that would change behavior and
defeat the point) — but it's the prime suspect, and Rung 1 will show us directly
whether it's hurting. This is a good example of *why you profile first*: it turns
a hunch into a measurement.

---

## 4. The working hypothesis

Putting it together: this is an **eager-mode, data-already-in-RAM, no-custom-kernel**
workload, with a per-step synchronization (`.item()`) and many small specialized
operations. So the most likely performance story is **not** "one slow kernel" —
it's **overhead and stalls**: time lost to launching many small kernels and to
the CPU/GPU waiting on each other. That's precisely what Rungs 1 and 2 diagnose,
and precisely why Rung 3 wouldn't help. The plan is shaped to test this
hypothesis cheaply.

---

## 5. What we'll actually do (and the safety rules)

We're profiling the **smallest representative run**: an MNIST config with one
small hidden layer, standard backprop, on a single GPU on a Slurm cluster (the
Kempner Institute's). Small = fast iteration while learning the tools, but it
still exercises the exact same training loop the big runs use.

The work is staged, with hard rules to keep it safe and honest:

1. **Opt-in only.** All profiling is gated behind an environment variable /
   command-line flag, **off by default.** With it off, training behaves exactly
   as before — same numbers, same speed. We are not changing the model or its
   results in any way.
2. **Tiny traces.** We profile only a *few steps after a warmup*, then stop —
   never the whole run. (Profilers generate huge files; a few steps is plenty.)
3. **Read between rungs.** After each tool, we stop and interpret the results
   together before deciding whether the next rung is worth it.
4. **Minimal, reviewable changes**, all on a side branch — never on the main
   branch.

### Rung 1, concretely (the immediate next step)
Add a small, flag-gated block around the training loop using `torch.profiler`. It
will record a handful of steps and produce two outputs:
- a **TensorBoard** trace (an interactive dashboard showing per-operation
  CPU/GPU time and a GPU-utilization summary), and
- a **Chrome trace** (a timeline you open in a browser at `chrome://tracing`).

We'll also add the **NVTX labels** mentioned in §2 at the same time, so they're
ready to make Rung 2's Nsight timeline readable later. The first question we'll
answer from Rung 1: **is the GPU busy, or is it idling — and do those `.item()`
calls show up as stalls?**

### Rung 2, later
An `nsys` (Nsight Systems) command run inside the cluster job, capturing CUDA,
NVTX, OS-runtime, and cuDNN/cuBLAS activity for a few steps. This gives the
full timeline: whether data loading overlaps with compute, how big the
between-kernel gaps are, and where synchronization points fall.

### Rung 3
Expected to be **skipped**, per the no-custom-kernels finding above — unless
Rungs 1–2 surprise us by pointing at a specific kernel we can influence.

---

## 6. Good questions to ask Claude about this

If you want to go deeper in chat, these are productive directions:
- "Explain what GPU utilization actually measures and why 100% isn't always the
  goal."
- "Why does calling `.item()` every step hurt, and what are the usual ways to
  avoid it without changing results?"
- "What would a *healthy* vs *unhealthy* torch.profiler result look like for this
  kind of workload?"
- "When IS Nsight Compute (Rung 3) worth it — what would have to be true?"
- "How do `num_workers`, `pin_memory`, and `prefetch_factor` interact, and how
  would I tell from a trace that data loading is the bottleneck?"
- "What's the difference between DDP and FSDP, and when would profiling each
  matter?"

---

## 7. One-paragraph summary

DendriNet is a single-framework **PyTorch** research codebase. Its training
inner loop is one well-defined function; its data sits **in memory** (so disk I/O
isn't the problem); and it has **no custom GPU kernels** (so the deepest
profiling tool, Nsight Compute, isn't warranted). The likely performance story is
**overhead and CPU/GPU stalls** — notably a per-step `.item()` synchronization —
rather than a single slow kernel. The plan is a three-rung "ladder": start with
`torch.profiler` to see if the GPU idles, climb to Nsight Systems for the full
timeline, and skip Nsight Compute. Everything is **opt-in, off by default, tiny,
and reviewed step by step**, with no change to the model's behavior or results.
