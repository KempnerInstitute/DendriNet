# Spatial Dendritic Cores for Pretrained Backbone Replacement

## 1. Problem Statement

We want to replace layers inside pretrained sequential CNN backbones (AlexNet, VGG) with dendritic excitatory-inhibitory (E-I) networks while keeping the rest of the backbone frozen. The goal is to study whether biologically-inspired dendritic computation can serve as a drop-in replacement for standard fully-connected or convolutional layers in vision models trained on ImageNet.

The existing `ConfigurableEINetwork` (DendriNet) operates on flat vectors — it takes `(B, D)` input and produces `(B, E)` output. Pretrained CNN backbones produce spatial feature maps `(B, C, H, W)`. We need infrastructure to bridge this gap, and eventually, a spatially-aware dendritic core that preserves locality.

**Scope**: The pretrained splitter (`pretrained.py`) currently supports models whose layers can be enumerated in forward-execution order — primarily AlexNet and VGG. Models with skip connections (e.g., ResNet) are supported only when the split points fall outside the residual blocks (e.g., at `avgpool` or `fc`). Broader backbone support is a future goal.

## 2. Architecture: The Existing DendriNet

Before describing the new work, here is how the existing DendriNet computes. This is the foundation both variants build on.

### 2.1 The Dendritic Tree

Each excitatory neuron (soma) has a hierarchical tree of dendritic branches. The tree is defined by **branch factors**. With `branch_factors = [3, 3]` and `n_soma = 4096`:

```
Level 2 (leaves):   4096 × 3 × 3 = 36,864 leaf branches
                         ↓ (BlockLinear: 3→1 weighted sum)
Level 1 (trunks):   4096 × 3 = 12,288 trunk branches
                         ↓ (BlockLinear: 3→1 weighted sum)
Level 0 (soma):     4096 somas
```

At each level, the branch equation is applied.

### 2.2 The Shunting Branch Equation

Each `DendriticBranchLayer` (one per tree level) receives three inputs:
1. **Feedforward input `x`** — the same input vector at every level (not just at the leaves)
2. **Inhibitory input** — from the inhibitory population, also the same at every level
3. **Child branch output** — the output of the previous (deeper) branch layer, or `None` at the leaves

This is a critical design property: **every dendritic level sees the original feedforward input**, not just the leaf branches. The hierarchy is not a simple bottom-up pipeline; each level integrates the feedforward signal alongside the child branch contributions. In `dendrinet.py`:

```python
def forward(self, x, inhibitory_input=None):
    output = None
    for i in range(self.n_branch_layers + 1):
        output = self.branch_layers[i](x, inhibitory_input, output)
    return output
```

At each level, the shunting equation (`branch_layer.py`) computes:

```
         E_ff + E_rec + branch_current(children)
V = ─────────────────────────────────────────────────────────────
     1 + E_ff + E_rec + I_ff + I_rec + g_branch(children) + ε
```

Where:
- **E_ff**: Excitatory feedforward drive from input `x`, via sparse TopK synapses. Enters both numerator (as current) and denominator (as conductance).
- **E_rec**: Excitatory recurrent drive (from previous temporal step, if recurrent). Same treatment.
- **branch_current(children)**: The `BlockLinear` layer computes a weighted sum of child branch outputs: `Σ_k w_k × V_child_k`. This is a current that enters the numerator.
- **g_branch(children)**: The `BlockLinear` layer also reports the sum of its weights via `sum_conductances()`. This total conductance enters the denominator.
- **I_ff**: Inhibitory feedforward drive. Enters the denominator only (divisive shunting).
- **I_rec**: Inhibitory recurrent drive. Enters the denominator only.
- **ε**: Small constant for numerical stability.

Note that `branch_current(children)` and `g_branch(children)` are distinct quantities from the BlockLinear — the current is the weighted sum of child voltages, while the conductance is the sum of the weights themselves. They are NOT the same term in numerator and denominator.

This is **shunting inhibition**: inhibition divides rather than subtracts. The output is bounded in `[0, 1)`.

After the shunting equation, a **reactivation** function is applied:

```
output = (tanh(m × (V - b)) + 1) / 2
```

where `m` (gain) and `b` (bias) are learnable parameters. This rescales the bounded shunting output.

### 2.3 Sparse Connectivity: TopKLinear

Each branch does NOT connect to all input neurons. Instead, **TopKLinear** implements sparse connectivity:

- Full weight matrix `W: (out_features, in_features)` is stored
- Only the top-K weights per output neuron are active (binary mask)
- Remaining weights exist in parameter space but are zeroed in the forward pass
- K is typically 10-40 synapses per branch (out of thousands of possible inputs)
- Weight transform (softplus/exp) ensures excitatory weights are positive
- Noise can be added during training for stochastic TopK selection

### 2.4 BlockLinear: Branch Convergence

When K child branches converge onto 1 parent branch, **BlockLinear** computes:

```
branch_current_i = Σ_k  w_ik × V_child_{i×K + k}     (enters numerator)
g_branch_i       = Σ_k  w_ik                           (enters denominator)
```

Each parent branch has K learnable weights (one per child). This is a block-diagonal linear map — neuron i's parent only sees neuron i's children, not other neurons' children.

### 2.5 E-I Layer Structure

Each `ExcitationInhibitionLayer` contains two populations:

- **E population**: DendriNet with `n_excitatory_cells` somas. Receives excitatory input + inhibitory feedback.
- **I population**: DendriNet (or MLP) with `n_inhibitory_cells` somas. Receives excitatory input, provides inhibitory output.

Forward pass: I cells fire first, then E cells receive I output as divisive inhibition.

Connectivity parameters per layer:
- `ee_synapses_per_branch`: E→E (feedforward excitation to E cells)
- `ei_synapses_per_branch`: E→I (feedforward excitation to I cells)
- `ie_synapses_per_branch`: I→E (inhibitory feedback)
- `ii_synapses_per_branch`: I→I (inhibitory self-regulation)

## 3. What Was Built: Variant 1 — Tensor Map Baseline (Control)

Variant 1 is the **baseline/control**. It validates the entire pretrained-replacement pipeline without introducing new dendritic math. The scientifically relevant target is Variant 2 (Section 4).

### 3.1 Overview

The tensor map is a thin wrapper that bridges spatial CNN encoders and the existing flat DendriNet:

```
Encoder output (B, C, H, W)
    → nn.Flatten → (B, C×H×W)
    → ConfigurableEINetwork → (B, E)
    → nn.Linear(E, suffix_dim) → (B, suffix_dim)    [only if E ≠ suffix_dim]
Decoder input (B, suffix_dim)
```

This reuses 100% of the existing DendriNet code. No new dendritic math was written.

### 3.2 Infrastructure Built

#### TensorSpec (pretrained.py)

A frozen dataclass that describes spatial tensor shapes:

```python
@dataclass(frozen=True)
class TensorSpec:
    channels: int
    height: int
    width: int
```

Added to `BackboneSplitPlan` as optional fields:
- `prefix_output_spec: TensorSpec | None` — spatial shape of encoder output before flattening
- `suffix_input_spec: TensorSpec | None` — spatial shape the suffix expects (if conv-based)

These are populated when `build_split_plan(..., spatial=True)` is called.

#### Spatial Split Path (pretrained.py)

`build_split_plan()` gained a `spatial: bool = False` parameter:

- When `spatial=True`, the prefix output shape is probed *without* auto-flatten to determine the raw `(C, H, W)` shape
- If the prefix contains both spatial and linear layers (e.g., AlexNet K=1 where prefix includes fc6), the spatial path gracefully falls back — `prefix_output_spec` stays None
- `build_encoder_decoder_from_plan()` uses `auto_flatten=not plan.is_spatial` for the encoder, so a spatial encoder outputs `(B, C, H, W)` instead of `(B, D)`

#### Spatial Decoder Guard (pretrained.py)

If a split plan has `suffix_input_spec` set (meaning the suffix contains conv layers that need spatial input), `build_encoder_decoder_from_plan()` raises `NotImplementedError`. This prevents a confusing shape error and documents clearly that spatial decoder construction is not yet supported. This guard will be removed when Variant 2 adds spatial decoder support.

#### HierarchicalDendriticTensorMap (tensor_map.py)

The core wrapper class. Key design decisions:
- Accepts both `(B, C, H, W)` and `(B, D)` input — flattens if needed
- Delegates all training hooks: `decay_weights()`, `apply_rewiring()`
- Delegates all introspection: `branch_layers`, `n_branch_layers`, `layers`, `get_effective_params()`
- Optional output projection when EINet output dim ≠ suffix input dim
- Registered in factory under aliases: `hierarchical_dendritic_tensor_map`, `dendritic_tensor_map`, `tensor_map`

#### Setup Utils Integration (setup_utils.py)

`_build_from_pretrained_replacement()` now:
- Detects spatial core types and passes `spatial=True` to `build_split_plan()`
- Passes `suffix_input_dim` to `get_architecture()` so the tensor map can add its output projection

### 3.3 Files Changed

| File | Change |
|------|--------|
| `pretrained.py` | Added `TensorSpec`, spatial fields on `BackboneSplitPlan`, `spatial` param on `build_split_plan`, spatial-aware `BackboneSegment`, decoder guard |
| `factory.py` | Added `_SPATIAL_DENDRITIC_TYPES`, `suffix_input_dim` param on `get_architecture`, tensor map dispatch |
| `setup_utils.py` | Auto-enables spatial mode for spatial core types, passes `suffix_input_dim` |
| `tensor_map.py` | New file — `HierarchicalDendriticTensorMap` wrapper |
| `spatial_dendritic/__init__.py` | New file — package init |
| `alexnet_k2_tensor_map.yaml` | New file — example config |
| `test_spatial_dendritic.py` | New file — 30 tests |

### 3.4 What the Tensor Map Does NOT Do

- **No spatial locality**: Every dendritic synapse can connect to any input element. A branch on soma 0 could connect to pixel (0,0) and pixel (223,223) simultaneously.
- **No conv replacement**: Cannot replace mid-network conv layers — only FC layers or entire FC blocks.
- **Parameter scaling**: Input dim is `C×H×W` (e.g., 9216 for AlexNet after avgpool), which means TopK synapses sample from a very large input space.

### 3.5 Test Coverage

30 tests covering:
- `TensorSpec` creation, properties, immutability
- Spatial vs. non-spatial split behavior on AlexNet K=1 and K=2
- Spatial encoder preserving 4D output
- Conv-suffix guard (`NotImplementedError`)
- Tensor map with flat and spatial input
- Output projection (with/without)
- Factory type aliases
- Training hook delegation (`decay_weights`, `apply_rewiring`)
- Introspection delegation (`n_branch_layers`, `layers`, `get_effective_params`)
- Gradient flow
- End-to-end pipeline: spatial encoder → tensor map → decoder on AlexNet

## 4. What Comes Next: Variant 2 — Hierarchical Dendritic Conv (Target Architecture)

Variant 2 is the **biologically relevant target architecture**. The tensor map baseline (Variant 1) serves as a control to isolate the effect of spatial locality from the effect of dendritic computation itself.

### 4.1 Motivation

The tensor map flattens spatial structure, which means:
1. A synapse connecting to position (0,0) has the same cost as one connecting to (5,5) — there is no spatial locality bias
2. The core cannot be used to replace conv layers (the suffix would need spatial input)
3. The parameter space is wasteful — most synapses will learn to connect to nearby positions anyway

Variant 2 solves this by constraining synapses to local spatial neighborhoods, exactly like a conv kernel, but with the full hierarchical dendritic tree and shunting inhibition.

### 4.2 Core Idea

There is one **shared** `ConfigurableEINetwork` (DendriNet) with `n_soma = C_out`. It is applied to every spatial position by unfolding the input into local patches and batching them together. The DendriNet's synapses are constrained to a local receptive field (the patch), not the global input.

**Existing DendriNet (flat)**:
- Each TopKLinear synapse samples from the *entire* input vector `(D,)`
- Synapse mask: top-K out of D possible connections

**Variant 2 (spatial)**:
- Each synapse samples from a *local patch* around position `(h, w)` in the input feature map
- The patch has size `(C_in, k_h, k_w)` — like a conv kernel's receptive field
- Synapse mask: top-K out of `C_in × k_h × k_w` possible connections within the patch
- Weights and masks are **shared across all spatial positions**, exactly like conv weight sharing

Implementation-wise, this is NOT one module per position or per channel. It is a single shared DendriNet evaluated on a batch of unfolded patches.

### 4.3 Architecture in Detail

For a layer replacing a `Conv2d(C_in, C_out, kernel_size=K, padding=P)`:

```
Input: (B, C_in, H, W)

1. Unfold into patches:
   patches = F.unfold(input, kernel_size=K, padding=P, stride=S)
   # shape: (B, C_in × K × K, H_out × W_out)

2. Reshape for batched DendriNet evaluation:
   patches = patches.permute(0, 2, 1).reshape(B × H_out × W_out, C_in × K × K)

3. Run shared DendriNet:
   output = dendrinet(patches, inhibitory_input)
   # shape: (B × H_out × W_out, C_out)

4. Reshape to spatial:
   output = output.reshape(B, H_out, W_out, C_out).permute(0, 3, 1, 2)
   # shape: (B, C_out, H_out, W_out)
```

The dendritic tree structure per soma is identical to the existing one:

```
        soma[c_out]
        /        |        \
    trunk_0   trunk_1   trunk_2          ← branch_factor[1] = 3
    / | \     / | \     / | \
  l0 l1 l2  l3 l4 l5  l6 l7 l8         ← branch_factor[0] = 3, total 9 leaves

Each leaf branch:
  - E synapse: TopK from flat_patch (K_ee out of C_in × k_h × k_w)
  - I synapse: TopK from inhibitory_pool (K_ie out of C_inh)
```

**Crucially**, the depth-wise signal flow inside the DendriNet is preserved exactly as in the current codebase: every branch layer receives:
1. The feedforward patch `x` (same at every dendritic level, not just leaves)
2. The inhibitory input (same at every level)
3. The child branch output from the previous dendritic depth

This means a trunk branch sees both the original patch AND the converged leaf output. It does not merely relay children — it integrates the feedforward signal at every depth. This property is what makes dendritic hierarchy meaningful, and the spatial variant must preserve it.

### 4.4 Relationship to Existing Components

| Component | Reused as-is? | What changes |
|-----------|---------------|--------------|
| Shunting equation | Yes | Same branch equation from `branch_layer.py` |
| Reactivation | Yes | Same `ParametricTanh` |
| BlockLinear | Yes | Same branch convergence |
| TopKLinear | Yes | Same module, but `in_features = patch_dim` instead of global dim |
| DendriNet | Yes | Same module, but `input_dim = patch_dim`, `n_soma = C_out` |
| ExcitationInhibitionLayer | Yes | Same module — E and I populations with `C_out` / `C_inh` somas |
| ConfigurableEINetwork | Yes | Same wrapper — constructs from config, same as tensor map |

The key insight: **no existing dendritic code needs modification**. The DendriNet already operates on flat vectors. We just feed it flat patches instead of flat global vectors, and batch spatial positions together.

### 4.5 Key Design Questions

**Q: Do all somas at different spatial positions share the same TopK mask?**

Yes — full weight sharing across positions, like conv. This is the correct analogue to convolution and the only practical starting point. Parameters: `O(C_out × tree_params)`, independent of spatial resolution.

**Q: How is the inhibitory population organized spatially?**

The I population mirrors the E population's spatial structure: `n_inhibitory_cells` channels at each spatial position. Each I cell has its own dendritic tree over the same local patch. I→E feedback is local: the I cells at position `(h, w)` inhibit the E cells at `(h, w)`. Since positions are batched together, this happens automatically.

**Q: How to handle the TopK synapse efficiently over patches?**

Use `F.unfold()` to extract all patches, then apply the existing `ConfigurableEINetwork` on the batched patches. Weight-shared means one set of TopKLinear weights applied across all positions. No new sparse operations needed.

### 4.6 Implementation Plan

**Initial scope**: Single `Conv2d` (or `Conv2d + activation`) replacement in AlexNet/VGG sequential backbones. Do not attempt arbitrary multi-layer conv ranges or ResNet residual blocks in v1.

#### New files:
- `spatial_dendritic/spatial_conv.py` — `HierarchicalDendriticConv` class

#### Modified files:
- `factory.py` — Register `hierarchical_dendritic_conv` type
- `spatial_dendritic/__init__.py` — Export new class
- `pretrained.py` — Remove `NotImplementedError` guard, support spatial decoder construction

#### The `HierarchicalDendriticConv` class:

```python
class HierarchicalDendriticConv(BaseNetwork):
    """Spatially local hierarchical dendritic core.

    Replaces a Conv2d layer (or Conv2d + activation) with dendritic E-I
    computation over local patches. A single shared ConfigurableEINetwork
    is evaluated on all unfolded spatial positions in a batched forward pass.

    The spatial contract:
      - input spec from prefix must be provided (C_in, H, W)
      - output spec must exactly equal suffix_input_spec (C_out, H_out, W_out)
    """

    def __init__(self, config, input_spec: TensorSpec,
                 kernel_size, stride=1, padding=0):
        # patch_dim = input_spec.channels × kernel_size × kernel_size
        # n_soma = excitatory_layer_sizes[-1] = C_out
        # Build a ConfigurableEINetwork with input_dim=patch_dim
        # Same config namespaces: architecture, connectivity, morphology, etc.
        # Small new 'spatial' block for kernel_size, stride, padding

    def forward(self, x):  # x: (B, C_in, H, W)
        # 1. Unfold: (B, patch_dim, H_out×W_out)
        # 2. Batch: (B×H_out×W_out, patch_dim)
        # 3. ConfigurableEINetwork forward → (B×H_out×W_out, C_out)
        # 4. Reshape: (B, C_out, H_out, W_out)
```

#### Config shape:

```yaml
core:
  type: "hierarchical_dendritic_conv"
  spatial:
    kernel_size: 3
    stride: 1
    padding: 1
  architecture:
    excitatory_layer_sizes: [256]     # C_out
    inhibitory_layer_sizes: [64]
    excitatory_branch_factors: [3, 3]
    inhibitory_branch_factors: [3, 3]
  connectivity:
    ee_synapses_per_branch_per_layer: [16]   # K out of C_in×k×k
    ei_synapses_per_branch_per_layer: [4]
    ie_synapses_per_branch_per_layer: [4]
    ii_synapses_per_branch_per_layer: [0]
  # ... rest identical to standard einet config (morphology, sparsity, etc.)
```

### 4.7 Recommended Implementation Constraints

1. **Start with single Conv2d replacement** (or Conv2d + activation). Do not try to infer geometry for arbitrary multi-layer conv ranges in v1.
2. **Explicit spatial contract**: input spec from prefix, output spec must exactly equal `suffix_input_spec`.
3. **Weight sharing across positions**: the correct conv analogue and only practical starting point.
4. **Reuse existing config namespaces** (architecture, connectivity, morphology). Add only a small `spatial` block for `kernel_size`, `stride`, `padding`.
5. **Target AlexNet/VGG first**. These are simple sequential models where conv layer replacement is straightforward. Broader backbone support comes later.
6. **Be aware of memory at early layers**: TopKLinear stores full `(out, in)` weight matrices. For early conv layers with large spatial dimensions, `B × H_out × W_out` can be very large.

### 4.8 Comparison Summary

| Property | Variant 1: Tensor Map (Control) | Variant 2: Spatial Conv (Target) |
|----------|--------------------------------|----------------------------------|
| Role | Baseline/control | Biologically relevant target |
| Input shape | `(B, C, H, W)` or `(B, D)` | `(B, C, H, W)` |
| Output shape | `(B, suffix_dim)` — flat | `(B, C_out, H_out, W_out)` — spatial |
| Spatial locality | None — synapses are global | Local receptive field (kernel_size) |
| Replaces | FC layers (classifier block) | Conv layers (feature extractor) |
| Synapse input space | Full flattened dim (e.g., 9216) | Local patch (e.g., 256×3×3 = 2304) |
| Suffix compatibility | Flat decoder (Linear) | Spatial decoder (Conv2d) |
| New dendritic code | None — 100% reuse | None — same DendriNet on unfolded patches |
| Feedforward at every depth | Yes (inherited from DendriNet) | Yes (must be preserved) |
| Status | **Complete** | **Not yet implemented** |

### 4.9 What Makes This Scientifically Interesting

Standard convolution computes:

```
y[c_out, h, w] = Σ_{c_in, i, j} W[c_out, c_in, i, j] × x[c_in, h+i, w+j] + b[c_out]
```

This is a weighted sum — linear, no E/I separation, no hierarchical structure, no divisive normalization.

The dendritic conv computes, per output position, at each dendritic depth `d`:

```
              E_ff(patch) + E_rec + branch_current(children_d)
V_d = ───────────────────────────────────────────────────────────────────
       1 + E_ff(patch) + E_rec + I_ff(patch) + I_rec + g_branch(children_d) + ε
```

Where every depth `d` receives the original local patch as feedforward input — the hierarchy is not a simple relay chain but a multi-level integration of the same signal at progressively coarser dendritic resolution.

This lets us test whether:
1. **Shunting inhibition** provides better feature normalization than BatchNorm
2. **Sparse E/I connectivity** can match dense conv performance with fewer active parameters
3. **Hierarchical dendritic processing** captures features that flat convolution cannot
4. **Replacing specific conv layers** (e.g., late feature extractor) with dendritic computation improves robustness or interpretability

The tensor map baseline isolates the dendritic computation effect from the spatial structure effect: if Variant 1 (global synapses, flat output) and Variant 2 (local synapses, spatial output) differ in performance, the difference is attributable to spatial locality, not to the dendritic equations themselves.
