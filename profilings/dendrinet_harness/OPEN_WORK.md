# Open work — optimization + scaling (as of 2026-07-30)

Four tracks. Written down because the state had stopped fitting in a conversation.

Cost key: **S** = minutes on one GPU · **M** = an allocation · **L** = multi-day / multi-GPU.

---

## 1. Kernel / feedforward speed

Today's net: forward gather **+8.6%** end-to-end (bit-identical), backward above-ceiling
**+11–20%**. Full detail in `ff_gather_roofline.md`.

| # | item | cost | notes |
|---|---|---|---|
| 1.1 | **Run `probe_shared_budget.py`** | S | Committed unrun. The wpo kernel was designed at IN=10,000 (40 KB → 2 blocks/SM → 66% occupancy); configs grew to IN=16,000 (64 KB → 1 block → 33%). If per-SM shared is 128 KB, 2 blocks may be a *near-miss* recoverable by carveout — **33%→66% on the staged path with no kernel change**. Highest value per minute of anything here. |
| 1.2 | Partial-row staging kernel | M | Only if 1.1 says 2 blocks is truly unreachable. Stage 32 KB → 4 blocks/SM, pay a second pass over idx/w. The file's own progression (stage less, three times) suggests it. |
| 1.3 | Re-measure the crossover rule on A100 | M | `_pick_fwd_variant` thresholds are **Blackwell-only**. A100's larger optin (166,912 B) makes the rule select `staged` — a no-op, not a regression, but unverified. |
| 1.4 | Sweep the picker across K | M | Only K=40 was swept. The `reuse = O·K/IN` term carries K on mechanism, not measurement. |

**Closed — do not re-litigate:** backward grad_x (all three routes measured: global atomics
0.47×, scatter→gather transpose 2.2–2.4× slower, staged wins — its 33% occupancy is a
*price* for atomic locality); backward tile P (1.00× weighted, and raising P is 17× worse);
forward tile P (P=4096 optimum, ~1.5% over shipped).

---

## 2. Python sweep — genuinely incomplete

Map: **15 of 488 files** execute per training step (`ff_stress/pymap/`). 3,887 aten ops/step.

| # | item | cost | notes |
|---|---|---|---|
| 2.1 | **`weight_transforms.py:119`** | S | **The big one, never examined.** 28.9% of live allocation; 132 ops + 132 `softplus` + 132 `softplus_backward` per step. Every claim I made about the Python layer being done was made *without* measuring this. |
| 2.2 | Per-step constant recomputation | S | `_load_ext()` **180×/step**, `cuda_gather_available()` 120×, `_pick_tile()` / `_optin_shared()` / `_pick_wpo_params()` / `_wpo_ok()` / `_lookup_transform()` / `_workspace_limited_chunk_size()` 60× each. All return values that cannot change between steps. Near-free on feedforward at batch 512; **this is the shape of thing that bites the batch-1 RNN**, ×T timesteps. |
| 2.3 | Fix the gather wrapper's dtype round trip | S | Measured: int32 → int64 → int32 → int32, costing 5.2% of the backward and ~386 MB of retained int64 that should be int32. Undoes part of the int32-index scaling lever. Small at this scale, **matters at 1M**. |
| 2.4 | Fix `audit_hotpath.py:212` and re-run B/D | S | `"cuda" if pd.assert_path_available and False else "fused"` is unconditionally `fused`. Passes B and D have **never** run on the path you'd actually use. |
| 2.5 | Decide on the sigmoid identity default | — | Shipped **off** (`DENDRINET_POSITIVE_TANH_SIGMOID`). Measured: 1.007× (fused) / 1.034× (cuda) speed, **1.19× memory = 4.8 GB at batch 512**, fp64 bit-identical, no drift over 200 Adam steps. It is a memory lever, and memory is the 1M blocker. Your call to flip it. |

---

## 3. RNN

| # | item | cost | notes |
|---|---|---|---|
| 3.1 | **Re-test path selection** | S | Configs run `fused`; cuda/triton **off**. That decision predates the adaptive backward tile. On feedforward, exactly this kind of stale flag was hiding **2.8×**. Cheapest high-upside check on the list. |
| 3.2 | Sigmoid identity on an RNN config | S | Untested there. The RNN forward is **31–37% GPU-busy across 13k kernels** — launch-bound — so removing 2 kernels per activation × T timesteps should pay far more than the 3.4% it gave on saturated feedforward. |
| 3.3 | Forward tile at RNN shapes | M | My "closed at 1.00×" sweep was batch 512, out=48k–144k. RNN is batch 1, out=1k–4k, where the grid collapses. **Different regime — my result does not transfer.** |
| 3.4 | CUDA-graph the cell (#383) | M | Capture verified working; **1.8–3.2× ceiling** measured on forward. `torch.compile` blocked two ways. |

---

## 4. Scaling to 1M / 10M

Ledger: `rnn_scaling_benchmark.md`. 100k PASS multi-GPU; 1M **OOM** on 4×A100-40GB.

| # | item | cost | notes |
|---|---|---|---|
| 4.1 | **Decompose the 13.75 GB per-GPU floor** | M | ~**11.9 GB unattributed** (activations + FSDP buffers). The ledger explicitly says do this **before** sharding idx. Fitting `mem(N)=F+K/N` gave F≈13.75 GB, of which replicated int32 idx is only 1.86 GB. |
| 4.2 | Model-axis scale differential | M | Never run. Everything measured so far is the **batch** axis. This is the pass that actually speaks to 10M. Needs a controlled series (K = 40/80/160, ≥3 points). |
| 4.3 | Does the sigmoid identity cut the 100k floor? | M | It returns 4.8 GB of *activation* memory at batch 512 — the same class as the hoist, aimed at the non-sharding term. Open row in the ledger. |
| 4.4 | 8-bit / bf16 optimizer state | M | Touches the **sharded** term only, so it helps less than 4.1 at fixed GPU count. |
| 4.5 | Shard the replicated idx buffer | M | 18.6 GB at 1M — necessary but **not sufficient**; alone it does not explain the floor. |
| 4.6 | Multi-node FSDP + comm overlap; parallelism search | L | The long pole for 10M. |

---

## 5. Figures / docs

| # | item | cost | notes |
|---|---|---|---|
| 5.1 | Regenerate the CUDA columns | M | Now stale by **~17%** (backward tile) **+ 8.6%** (forward variant) — they understate the cuda path substantially. Three sets, both GPUs. |

---

## UNVERIFIED — run these first on the next allocation

Two changes are committed but were **never executed on a GPU** (the allocation expired
between writing and testing). They are small and mechanical, but "should be right" is not
a measurement:

| # | change | what to run |
|---|---|---|
| U.1 | **C6** — gather wrapper now stores indices as int32 instead of upcasting to int64 and retaining the larger copy | `check_gradw_tile_equiv.py` (must PASS), then `bench_gather_variant_e2e.py` — expect ~386 MB less peak, output still bit-identical |
| U.2 | **C5** — `_pick_tile()` memoized (was querying `get_device_capability()` 60×/step) | same runs; expect **~0** on feedforward. Only interesting on the batch-1 RNN, where host work is paid T times per sequence |

If U.1 regresses anything, revert it first — it touches the gather hot path, unlike
everything else in the reactivation work.

## Recommended order

1. **2.1** (`weight_transforms.py`) — largest thing never measured, and the last leg of the claim I over-stated.
2. **1.1** (`probe_shared_budget.py`) — one minute, could hand back 33%→66% occupancy for free.
3. **3.1** (RNN path A/B) — one stale flag already cost 2.8× once.
4. **4.1** (the 13.75 GB floor) — the actual 1M blocker; everything else in track 4 waits on it.

2.2–2.4 are cheap and can ride along with any allocation.
