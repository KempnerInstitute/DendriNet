# Python-level optimization candidates (2026-07-30)

Hunting for another win shaped like the fused-gather autograd function. That one won by
**refusing to materialize a large intermediate** — it recomputed in backward instead of
storing. 3.4× speed but **7.5× memory**, and the memory is what turned an OOM into a
runnable config.

So the pattern to look for is not "slow function". It is:

> a full-size tensor that is a pure function of something already in memory, materialized
> in forward and retained for backward.

Search space is the **15 files** that actually execute per step (`ff_stress/pymap/`).

**Evidence key:** **[M]** measured · **[P]** visible in a profile but unquantified ·
**[S]** suspected from reading the code, unmeasured.

## Which gather path do these assume?

Two different provenances are mixed below, so this has to be stated up front.

- The **aten-op map** (3,887 ops/step, and every op count quoted here) was measured on the
  **cuda** path — `map_ff_python.py --path cuda`. Sound.
- The **allocation profile** — including C1's headline **28.9%** — came from pass D, which
  had a bug (`"cuda" if ... and False else "fused"`) forcing it to **`fused` at batch 64**.
  Fixed 2026-07-30; `--path` is now explicit and required in practice.

The absolute allocation from C1 is unchanged by path, but the *share* is not: `fused` and
`cuda` retain different things for backward, so the denominator moves. **Re-run pass D on
`--path cuda --batch 512` before quoting 28.9% again.**

Which candidates depend on the path at all:

| candidate | path dependence |
|---|---|
| C1 | **none.** `_transformed_weight()` runs at `indexed_sparse.py:501`, *before* path selection at :514 — it materializes on loop, fused, triton and cuda alike |
| C2, C3, C4 | **none** — BlockLinear, module-attribute retention, and branch dynamics are all outside the gather |
| C5 | **split.** `_load_ext()` 180× and `cuda_gather_available()` 120× are **cuda-only** (`and` short-circuits when `use_cuda_gather` is off); `_lookup_transform()` 132× and `_forward_output_chunk_size()` 60× are path-independent |
| C6 | **cuda-only** by definition |

So the list does not assume the fused autograd gather is on. C1 — the one to do first —
holds on **every** path including the naive loop.

---

---

# MEASURED RANKING (cuda, batch 512, Blackwell — 2026-07-30)

`profile_c1_weight_transform.py`, live allocation at the top of the step (after forward,
before backward frees anything). Peak 28,184 MB, live 25,936 MB.

| MB | share | site | was |
|---:|---:|---|---|
| **14,943.5** | **57.6%** | `parametric.py:57` `_positive_tanh_centered` | not on the list |
| 4,559.8 | 17.6% | `branch_dynamics.py:285` | C4 |
| 3,432.8 | 13.2% | `branch_dynamics.py:213` | C4 |
| 1,550.6 | 6.0% | `branch_dynamics.py:245` | C4 |
| 890.3 | 3.4% | `cuda_gather.py:619` forward | — |
| **448.6** | **1.7%** | `weight_transforms.py:119` | **C1** |

**C1 is dead: 1.7% of live, 1.6% of peak — not 28.9%.** The old figure was exactly the
fused/batch-64 artifact predicted. Arithmetic cross-check passed (443.1 MB predicted vs
448.6 measured, AGREE), so the attribution is trustworthy and this is a real refutation
rather than a broken measurement.

**The real #1 is the reactivation** at 57.6% — and `parametric.py:57` is the *slow* branch,
taken because `DENDRINET_POSITIVE_TANH_SIGMOID` defaults off. The identity already built and
verified today measured **1.19× peak end-to-end (4.8 GB)** against precisely this line.

**The real #2 is C4**, `branch_dynamics` accumulation, at **36.8% combined** — far larger
than the "[P] 120 ops" estimate that had it fourth.

### DONE — the reactivation, 28,184 → 18,551 MB peak (1.52×, 9.6 GB)

Both steps shipped and measured end-to-end (cuda, batch 512):

| mode | step (ms) | peak (MB) | vs eager | saved |
|---|---|---|---|---|
| eager `(tanh(m(x-b))+1)/2` | 852.8 | 28,184.1 | 1.00× | — |
| identity `sigmoid(2m(x-b))` | 819.2 | 23,384.7 | 1.21× | 4,799 MB |
| **fused (retains only y)** | 841.8 | **18,551.5** | **1.52×** | **9,633 MB** |

Output deviation 2.39e-07, grad_x 3.97e-07 — rounding, and the fused path's gradient gap
is *smaller* than the identity's.

The mechanism was confirmed exactly. Eager retains **three** full-size tensors (`x-b` for
MulBackward, `tanh(...)` for TanhBackward, and the output); fused retains **one**. The
line went 14,943.5 → 4,980.4 MB = **3.0×**, precisely 3→1.

Honest trade: fused is **2.7% slower than identity** (841.8 vs 819.2 ms) because its
backward computes `logit(y)`. Still faster than eager, and it buys another 4.8 GB. Memory
is the 1M blocker, so that is the right side of the trade.

### Revised order

1. ~~sigmoid identity~~ — **done**, on by default (`1d1b1159`).
2. ~~`parametric.py` structural~~ — **done** (`05063123`).
3. ~~C4 (`branch_dynamics`)~~ — **done** (`_ShuntDividePositiveTanh`).

### C4 DONE — fused shunt divide, another 4.2 GB

The accumulation chain was a red herring: `a + b` retains nothing for backward, so every
intermediate is freed as it is superseded. The cost was the **divide** — `DivBackward` pins
both operands, so `voltage = n / (d + eps)` held the numerator *and* the `d+eps` it
allocated, while the activation held `y`. Measured: **2.92 full-size tensors** against a
one-tensor cost of 4,968.8 MB.

Fusing the divide into the reactivation that consumes it (same function, line 323) drops
that to 2, because the numerator is never needed — `dL/dn = dL/dv/(d+eps)` — and `v` is
reconstructible from `y` as `c + logit(y)/(2m)`, the same trick as `z`, resting on the same
saturation argument.

**Confirmed by re-profiling: 2.92 → 1.92 tensors**, and the three `branch_dynamics` lines
now hold **0.00 MB** — the allocation moved inside the fused Function.

### Cumulative, 12L/16k stress config, cuda, batch 512

| mode | step (ms) | peak (MB) | vs eager | saved |
|---|---|---|---|---|
| eager | 857.7 | 28,184.1 | 1.00× | — |
| + identity | 820.5 | 23,384.7 | 1.21× | 4,799 MB |
| + fused activation | 841.1 | 18,551.5 | 1.52× | 9,633 MB |
| **+ fused shunt** | 844.8 | **14,343.3** | **1.96×** | **13,841 MB** |

Output deviation 2.39e-07 throughout; grad_x 7.08e-07. Step time stays *below* eager.

Equivalence (`check_fused_shunt.py`): output bit-identical; in **fp64 every gradient — n, d,
log_m, b — agrees to ~1e-16**, i.e. machine epsilon, so the derivation is exact rather than
approximate. Guarded off whenever diagnostics or analysis-current capture need `voltage`
itself, and only ever applied to the positive-tanh family.

4. **Next: C5, C6, C3 as cheap ride-alongs.** New ranking (live ≈ 11.0 GB):

   | MB | share | site |
   |---:|---:|---|
   | 4,980.3 | 45.3% | `parametric.py:173` — `y` |
   | 4,559.4 | 41.5% | `parametric.py:171` — `d+eps` |
   | 890.0 | 8.1% | `cuda_gather.py:619` |
   | 449.1 | 4.1% | `weight_transforms.py:119` (C1, closed) |

   Both remaining big items are now irreducible for a divide-then-activate: any two of
   `{n, d+eps, v}` suffice and two are required. Further cuts need a different idea
   (recompute across the layer boundary, or checkpointing), not another fusion.

5. **C1, C2, C3, C7 are closed.**

---

## C1 — `sparse_weight()` materializes the transformed weight — **REFUTED, 1.6% of peak**

`indexed_sparse.py:293` → `weight_transforms.py:119` → `softplus(pre_w)`

- **28.9% of live allocation** (pass D), 132 ops + 132 `softplus` + 132 `softplus_backward`
  per step.
- Produces a full `[out_features, K]` tensor **and** `softplus` saves its input for
  backward, so *two* full-size tensors are live per layer. At out=144,000 / K=40 that is
  ~23 MB × 2 × 60 layers.
- It is a **pure elementwise function of a parameter** — the exact precondition the fused
  gather exploited.
- **Fix shape:** fold the transform into the gather. The kernel already reads
  `weight[base+k]`; have it read `pre_w[base+k]` and apply softplus inline, recomputing in
  backward. Eliminates the intermediate *and* its backward copy.
- **Why it is first:** biggest measured allocation never examined, and structurally
  identical to the win we are trying to repeat. Also sets up the Triton/CUDA phase.

## C2 — `EfficientBlockLinear.forward` — **REFUTED at the retained level, 8.9 MB**

Measured `blocklinear.py:313` at **8.9 MB — 0.05% of live allocation**, both before and
after the reactivation fix. Not 72-ops-worth of anything.

Why the guess was wrong: `(x_reshaped * weights).sum(dim=-1)` does build a
`[B, out, block]` product, but that tensor is **transient** — `sum` consumes it
immediately and `SumBackward` retains nothing but a shape. What survives into backward is
`x_reshaped` and `weights`, which are small here because this BlockLinear is the
branch→soma pooling with `block_size = 3`, not one of the 144,000-wide sparse layers.

**Caveat, stated because it is the one thing this profile cannot see:** the snapshot is
taken at the top of the step and therefore shows only *retained* allocation. Live total
was 15,969 MB against a step peak of 18,469 MB — a **~2.5 GB gap that is transients**, and
a large short-lived product would live in exactly that gap. C2 is refuted as a *retention*
problem; it is not yet ruled out as a contributor to peak. Decomposing that gap is the
honest follow-up.

The `apply_weight_transform(...) * 1.0` on the same line remains a trap either way — verify
whether it exists to force a non-leaf for `register_hook` before touching it.

## C3 — `_last_forward_param_tensor` — **REFUTED, it does not duplicate anything**

It never allocates; it stores a *reference* to a tensor allocated at
`weight_transforms.py:119`. So it cannot appear as its own line — it can only show up by
inflating that one.

It does not. `weight_transforms.py:119` measured **448.6 MB** against an arithmetic
prediction of **443.1 MB** for exactly *one* copy of the transformed weights across all 60
layers. One copy, not two — so the reference is aliasing the tensor autograd already holds,
not adding a second.

Its real (small) cost is lifetime, not size: the weight stays reachable between steps
instead of being freed. Bounded by 443 MB, and ~0 at peak, since peak occurs mid-step when
that tensor is live regardless. Not worth a conditional.

## C2 (original hypothesis, kept for the record) **[P]**

`blocklinear.py:317` — `output = (x_reshaped * weights).sum(dim=-1)`

- Allocates `[B, out_features, block_size]` and immediately reduces the last axis. This is
  **precisely** the multiply-then-sum the fused gather removed.
- 72 aten ops at that line; 36 each at :311 and :313.
- **Fix shape:** a fused multiply-reduce autograd Function (or `einsum`/`bmm`) that never
  builds the product.
- Also on that line: `apply_weight_transform(...) * 1.0`. It **looks** like a pointless
  full-size copy, but it may exist to force a non-leaf tensor for `register_hook` at :246.
  **Verify intent before removing** — this is a trap, not a free win.

## C3 — `_last_forward_param_tensor` pins the transformed weight **[S]**

`indexed_sparse.py:505`, `blocklinear.py:224` and `:316`

- Stores a strong reference to the transformed weight on the module, so it survives the
  whole step and until the next forward — on top of whatever autograd retains.
- Read **only** by `register_forward_param_gradient_scale` (`indexed_sparse.py:553`,
  `blocklinear.py:246`). When dynamic grad scaling is off, it is pure retention.
- **Upgraded to [P]:** no config under `configs/` enables dynamic grad scaling
  (`grep` for `dynamic_grad_scal|param_gradient_scale` returns nothing), so on every
  config you actually run this reference is held for a consumer that never fires.
- ~1.4 GB at the stress config if it is genuinely unused there.
- **Fix shape:** store only when the scaling feature is enabled. One conditional.
- Cheapest item on the list; verify the feature is actually off in your configs first.

## C4 — `branch_dynamics` accumulation chains **[P]**

`branch_dynamics.py:213–285`

- `shunt_numerator = shunt_numerator + …` and `shunt_denominator = shunt_denominator + …`
  run up to 5 times each, every one allocating a full `[batch, branches]` temporary, then
  `voltage = shunt_numerator / (denominator + epsilon)` adds two more.
- :285 = 120 ops (3.1%); :213/:214/:245/:246/:261 = 24–36 each.
- **Fix shape:** in-place accumulation where the operand is not needed for backward, or a
  single fused autograd Function for the whole shunting expression — which is also a clean
  Triton target later.

## C5 — Per-step recomputation of constants **[M]**

- `_load_ext()` **180×/step**, `cuda_gather_available()` 120×, and `_pick_tile()`,
  `_optin_shared()`, `_pick_wpo_params()`, `_wpo_ok()`, `_lookup_transform()`,
  `_forward_output_chunk_size()` / `_workspace_limited_chunk_size()` 60× each.
- Every one returns a value that cannot change between steps.
- Pure CPU, so it is nearly free on saturated feedforward at batch 512 — but the batch-1
  RNN is **31–37% GPU-busy across 13k kernels**, i.e. mostly CPU gap, ×T timesteps.
- **Fix shape:** memoize. Expect ~0 on feedforward and possibly real on the RNN. Measure on
  both; do not report the feedforward number as the verdict.

## C6 — Gather wrapper dtype round trip **[M]**

`cuda_gather.py:418–443` — indices go int32 → int64 → int32 → int32.

- Measured: **5.2% of the backward** and **~386 MB** of retained int64 that should be int32.
- Partly undoes the int32-index scaling lever. Small here, **matters at 1M**.

## C7 — `_normalized_sparse_weight` temporaries — **CLOSED, not a candidate**

`indexed_sparse.py:297`. When `weight_norm_order` is set this adds four more full-size
`[O, K]` tensors (`norm`, `safe_norm`, `normalized`, `torch.where`) on top of C1.

**But every config that mentions it sets `weight_norm_order: null`**, and the function
returns immediately on that branch. Zero cost on anything you run. Closed before it cost an
allocation — which is the point of checking liveness first.

## C8 — `indices.to(flat_x.device)` per forward **[S]**

`indexed_sparse.py:507` — a no-op returning `self` when already on device, so probably free.
Listed only so it gets ruled out by measurement rather than assumed.

---

## Method (so results stay trustworthy)

For each candidate, in order:

1. **Confirm it is live** in the config being measured — several are behind flags.
2. **Measure the isolated op** (allocation + time).
3. **Measure end-to-end on the `cuda` path**, not `fused`. The isolated number will
   overstate; only end-to-end counts.
4. **Prove equivalence** — fp64 bit-identity where the change is algebraic, tolerance +
   a drift run where it is not.
5. Record the number even when it is a loss.

Two traps already paid for today: an isolated 1.43× became 1.007× end-to-end (Amdahl), and
`fused` is 2.8× slower than `cuda`, so anything baselined on it looks better than it is.
