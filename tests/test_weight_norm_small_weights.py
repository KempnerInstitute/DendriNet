"""Synaptic weight normalization for rows below machine epsilon.

Rows whose norm is at least machine epsilon keep the historical expression bit
for bit: values, reverse-mode gradients and forward-mode tangents, and every
row for orders ``p <= 0``. Smaller rows (including subnormal weights and L2
norms that underflow) normalize to the precision of their dtype, with finite
gradients that match an independent float64 computation. ``exp`` and
``softplus`` rows depend only on differences of their parameters; rows without
contacts or nonzero weights pass through unchanged.
"""

import math

import mpmath
import pytest
import torch
from torch.nn import functional as F

from dendritic_modeling.analysis.utils.effective_synapses import (
    effective_synapse_snapshot,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_dynamic import (
    IndexedDynamicTopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.utils.weight_normalization import normalize_weight_rows
from dendritic_modeling.networks.utils.weight_transforms import apply_weight_transform

GAMMA = 2.0
KINDS = ["indexed", "topk", "dynamic"]
SMALL_ROW = 1
ZERO_ROW = 2


def _layer(kind, transform, order, dtype):
    torch.manual_seed(0)
    options = {"weight_transform": transform, "weight_norm_order": order}
    if kind == "indexed":
        layer = IndexedSparseLinear(12, 5, 4, gamma=GAMMA, **options)
    elif kind == "topk":
        layer = TopKLinear(12, 5, 4, gamma=GAMMA, **options)
    else:
        layer = IndexedDynamicTopKLinear(12, 5, 4, gamma=GAMMA, **options)
    return layer.to(dtype).eval()


def _views(layer):
    """Every accessor that returns normalized weights, as ``(name, tensor)``."""
    views = [
        ("pruned_weight", layer.pruned_weight()),
        ("snapshot", effective_synapse_snapshot(layer, detach=False).effective_weight),
    ]
    for name in ("_normalized_sparse_weight", "dense_weight"):
        if hasattr(layer, name):
            views.append((name, getattr(layer, name)()))
    return views


def _materialized(layer):
    """The weights each accessor normalizes, built with the layers' own ops."""
    weight = apply_weight_transform(layer.pre_w, layer.weight_transform)
    if isinstance(layer, TopKLinear):
        pruned = layer.weight_mask() * weight
        return {"pruned_weight": pruned, "snapshot": pruned}
    if isinstance(layer, IndexedDynamicTopKLinear):
        weight = weight * layer.active_sparse_mask()
    indices = layer.connection_indices.long()
    dense = torch.zeros(layer.out_features, layer.in_features, dtype=weight.dtype)
    dense = dense.scatter_add(1, indices, weight)
    views = {"_normalized_sparse_weight": weight, "pruned_weight": dense}
    active = effective_synapse_snapshot(layer).active_mask
    views["snapshot"] = layer.weight() * active
    if isinstance(layer, IndexedSparseLinear):
        views["dense_weight"] = dense
        # The snapshot normalizes the already normalized dense view again.
        views["snapshot"] = _historical(dense, layer.weight_norm_order, GAMMA) * active
    return views


def _historical(weight, order, gamma):
    norm = torch.norm(weight, p=order, dim=-1, keepdim=True)
    safe_norm = norm.clamp_min(torch.finfo(weight.dtype).eps)
    return torch.where(norm > 0, (weight / safe_norm) * gamma, weight)


def _small_values(transform, shape, dtype):
    """One row of positive weights whose norm lies below machine epsilon."""
    offsets = -0.5 * torch.arange(shape[-1], dtype=torch.float64)
    if transform in {"exp", "softplus"}:
        return (-90.0 + offsets).to(dtype)
    tiny = torch.finfo(dtype).tiny
    return (tiny * 16 * torch.exp(offsets)).to(dtype)


def _fill_rows(layer, transform):
    """Ordinary random rows, one small row and one row of zero weights.

    ``exp`` and ``softplus`` weights of -1000 underflow to zero, but their row
    is a small row normalized from its parameters, not a zero row.
    """
    with torch.no_grad():
        layer.pre_w[SMALL_ROW] = _small_values(
            transform, layer.pre_w.shape, layer.pre_w.dtype
        )
        layer.pre_w[ZERO_ROW] = -1000.0 if transform in {"exp", "softplus"} else 0.0


def _bits(tensor):
    return tensor.detach().view(
        torch.int64 if tensor.dtype == torch.float64 else torch.int32
    )


class _Apply(torch.nn.Module):
    """Runs ``fn(layer)`` so torch.func can substitute the layer's parameters."""

    def __init__(self, layer, fn):
        super().__init__()
        self.layer, self.fn = layer, fn

    def forward(self):
        return self.fn(self.layer)


def _forward_tangent(layer, fn, tangent):
    module = _Apply(layer, fn)
    return torch.func.jvp(
        lambda pre_w: torch.func.functional_call(module, {"layer.pre_w": pre_w}, ()),
        (layer.pre_w.detach(),),
        (tangent,),
    )[1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("order", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("transform", ["exp", "softplus", "identity", "relu"])
@pytest.mark.parametrize("kind", KINDS)
def test_rows_above_machine_epsilon_keep_historical_values_and_gradients(
    kind, transform, order, dtype
):
    layer = _layer(kind, transform, order, dtype)
    _fill_rows(layer, transform)
    # Orders p <= 0 are not norms and keep the historical expression throughout.
    keep = torch.ones(layer.out_features, dtype=torch.bool)
    keep[SMALL_ROW] = order <= 0
    log_space = transform in {"exp", "softplus"}
    keep[ZERO_ROW] = order <= 0 or not log_space
    # Negative orders of subnormal weights had non-finite gradients already.
    finite = torch.ones(layer.out_features, dtype=torch.bool)
    finite[SMALL_ROW] = order >= 0
    reference = _materialized(layer)
    for name, actual in _views(layer):
        expected = _historical(reference[name], order, GAMMA)
        assert torch.equal(_bits(actual[keep]), _bits(expected[keep])), name
        assert torch.isfinite(actual).all(), name

        tangent = torch.randn_like(layer.pre_w)
        actual_tangent = _forward_tangent(
            layer, lambda swapped, name=name: dict(_views(swapped))[name], tangent
        )
        expected_tangent = _forward_tangent(
            layer,
            lambda swapped, name=name: _historical(
                _materialized(swapped)[name], order, GAMMA
            ),
            tangent,
        )
        assert torch.equal(_bits(actual_tangent[keep]), _bits(expected_tangent[keep]))

        upstream = torch.randn_like(actual)
        upstream[:, ::3] = 0.0
        upstream[:, 1::4] = -0.0
        (actual_grad,) = torch.autograd.grad(
            actual, layer.pre_w, upstream, retain_graph=True
        )
        (expected_grad,) = torch.autograd.grad(
            expected, layer.pre_w, upstream, retain_graph=True
        )
        assert torch.isfinite(actual_grad[finite]).all(), name
        assert torch.equal(_bits(actual_grad[keep]), _bits(expected_grad[keep])), name
        if not log_space:
            assert torch.count_nonzero(actual[ZERO_ROW]) == 0, name


@pytest.mark.parametrize("order", [-2, -1, float("-inf")])
def test_negative_orders_keep_finite_historical_weights(order):
    # Masked contacts make these "norms" zero; the historical expression then
    # returns the selected weights unchanged.
    torch.manual_seed(0)
    layer = TopKLinear(4, 2, 2, weight_transform="exp", weight_norm_order=order)
    weights = layer.pruned_weight()
    selected = layer.weight_mask() * layer.weight()
    assert torch.equal(weights, _historical(selected, order, layer.gamma))
    assert torch.isfinite(weights).all()
    assert torch.isfinite(layer(torch.randn(3, 4))).all()


def _reference_weights(layer, pre_w):
    """Independent float64 normalization of the realized weights."""
    pre_w = pre_w.double()
    if layer.weight_transform == "exp":
        weight = torch.exp(pre_w)
    else:
        weight = F.softplus(pre_w)
    if isinstance(layer, TopKLinear):
        weight = weight * layer.weight_mask().double()
    elif isinstance(layer, IndexedDynamicTopKLinear):
        weight = weight * layer.active_sparse_mask().double()
    norm = torch.linalg.vector_norm(weight, ord=layer.weight_norm_order, dim=-1)
    return GAMMA * weight / norm[:, None]


@pytest.mark.parametrize(
    "dtype,level",
    [
        (torch.float32, -20.0),  # norm below eps
        (torch.float32, -60.0),  # squared weights underflow
        (torch.float32, -90.0),  # subnormal weights
        (torch.float16, -12.0),  # subnormal half-precision weights
        (torch.bfloat16, -90.0),
        (torch.float64, -40.0),
    ],
)
@pytest.mark.parametrize("order", [1, 2, math.inf])
@pytest.mark.parametrize("transform", ["exp", "softplus"])
@pytest.mark.parametrize("kind", KINDS)
def test_small_rows_normalize_exactly_with_reference_gradients(
    kind, transform, order, dtype, level
):
    layer = _layer(kind, transform, order, torch.float32)
    with torch.no_grad():
        offsets = -0.25 * torch.arange(layer.pre_w.shape[-1], dtype=torch.float32)
        layer.pre_w.copy_(level + offsets + 0.1 * torch.arange(5)[:, None])
    layer = layer.to(dtype)
    if isinstance(layer, IndexedSparseLinear):
        weights = layer._normalized_sparse_weight()
    else:
        weights = (
            layer.pruned_weight()
            if kind == "topk"
            else layer._normalized_sparse_weight()
        )
    pre_w64 = layer.pre_w.detach().double().requires_grad_()
    expected = _reference_weights(layer, pre_w64)

    tolerance = {
        torch.float16: 4e-3,
        torch.bfloat16: 3e-2,
        torch.float32: 2e-6,
        torch.float64: 1e-12,
    }[dtype]
    torch.testing.assert_close(
        weights.double(), expected, rtol=tolerance, atol=tolerance * GAMMA
    )
    norms = torch.linalg.vector_norm(weights.double(), ord=order, dim=-1)
    torch.testing.assert_close(
        norms, torch.full_like(norms, GAMMA), rtol=tolerance, atol=0
    )

    coefficients = torch.linspace(-1.0, 2.0, weights.numel()).reshape(weights.shape)
    (gradient,) = torch.autograd.grad(
        (weights * coefficients.to(dtype)).sum(), layer.pre_w
    )
    (expected_gradient,) = torch.autograd.grad(
        (expected * coefficients.double()).sum(), pre_w64
    )
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(
        gradient.double(),
        expected_gradient,
        rtol=10 * tolerance,
        atol=10 * tolerance * expected_gradient.abs().max().item(),
    )


@pytest.mark.parametrize("transform", ["exp", "softplus"])
def test_subnormal_double_rows_match_extended_precision(transform):
    layer = _layer("indexed", transform, 2, torch.float64)
    values = [-720.0, -721.5, -723.0, -726.0]
    with torch.no_grad():
        layer.pre_w.copy_(torch.tensor(values, dtype=torch.float64))
    weights = layer._normalized_sparse_weight()

    with mpmath.workdps(40):
        if transform == "exp":
            raw = [mpmath.exp(value) for value in values]
        else:
            raw = [mpmath.log1p(mpmath.exp(value)) for value in values]
        norm = mpmath.sqrt(sum(value**2 for value in raw))
        expected = [float(GAMMA * value / norm) for value in raw]
    for row in weights:
        torch.testing.assert_close(
            row, torch.tensor(expected, dtype=torch.float64), rtol=1e-13, atol=0
        )


@pytest.mark.parametrize("transform", ["identity", "relu"])
@pytest.mark.parametrize("order", [1, 2])
def test_small_signed_and_rectified_rows_normalize_exactly(transform, order):
    layer = _layer("indexed", transform, order, torch.float32)
    values = torch.tensor([3e-40, -1e-40, 2e-41, 5e-42])
    if transform == "relu":
        values = values.abs()
    with torch.no_grad():
        layer.pre_w.copy_(values)
    weights = layer._normalized_sparse_weight()
    expected = values.double() / torch.linalg.vector_norm(values.double(), ord=order)
    for row in weights:
        torch.testing.assert_close(row.double(), GAMMA * expected, rtol=2e-6, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_log_space_rows_depend_only_on_parameter_differences(kind):
    offsets = -2.0 * torch.arange(4, dtype=torch.float32)
    rows = {}
    for transform, levels in (
        ("exp", (0.0, -90.0, -1000.0)),
        ("softplus", (-40.0, -1000.0)),
    ):
        for level in levels:
            layer = _layer(kind, transform, 2, torch.float32)
            with torch.no_grad():
                layer.pre_w.fill_(-5000.0)
                layer.pre_w[:, :4] = level + offsets
            weights = layer.pruned_weight()
            (gradient,) = torch.autograd.grad(weights.square().sum(), layer.pre_w)
            assert torch.isfinite(gradient).all()
            rows[transform, level] = weights
    # Weights of -1000 underflow, yet the rows match unshifted parameters.
    for transform, levels in (
        ("exp", (0.0, -90.0, -1000.0)),
        ("softplus", (-40.0, -1000.0)),
    ):
        for level in levels[1:]:
            torch.testing.assert_close(
                rows[transform, level], rows[transform, levels[0]], rtol=2e-6, atol=0
            )
    # Entries whose weights underflow keep their exact share of the row.
    expected = GAMMA * torch.exp(offsets.double()) / torch.exp(offsets.double()).norm()
    underflowed = rows["exp", -1000.0]
    shares = underflowed[underflowed != 0].reshape(-1, 4).double()
    for row in shares.sort(dim=-1, descending=True).values:
        torch.testing.assert_close(row, expected, rtol=2e-6, atol=0)


def test_rows_without_contacts_pass_through():
    pre_w = torch.full((2, 3), -200.0, requires_grad=True)
    contacts = torch.tensor([[True, True, False], [False, False, False]])
    weight = torch.exp(pre_w) * contacts
    output = normalize_weight_rows(
        weight, 1, GAMMA, pre_weight=pre_w, transform="exp", contacts=contacts
    )
    torch.testing.assert_close(output, torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    (gradient,) = torch.autograd.grad(output.sum(), pre_w)
    assert torch.isfinite(gradient).all()


def test_rows_of_exact_zeros_return_the_input_and_its_gradient():
    weight = torch.zeros(2, 3, requires_grad=True)
    upstream = torch.tensor([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]])
    output = normalize_weight_rows(weight, 2, GAMMA)
    assert torch.equal(output, weight.detach())
    (gradient,) = torch.autograd.grad(output, weight, upstream)
    assert torch.equal(gradient, upstream)


@pytest.mark.parametrize("level", [-1.0, -40.0])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("transform", ["exp", "softplus", "identity"])
def test_gradcheck_including_forward_mode_double_backward_and_vmap(
    transform, order, level
):
    mask = torch.tensor(
        [[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]], dtype=torch.float64
    )

    # Identity weights are scaled to the level inside the function, so the
    # finite differences of gradcheck stay on one side of the L1 kink at 0.
    scale = math.exp(level) if transform == "identity" else 1.0

    def normalize(raw):
        pre_w = raw * scale
        weight = apply_weight_transform(pre_w, transform) * mask
        return normalize_weight_rows(
            weight, order, GAMMA, pre_weight=pre_w, transform=transform
        )

    steps = 0.3 * torch.arange(8, dtype=torch.float64).reshape(2, 4)
    raw = 1.0 + steps if transform == "identity" else level + steps
    raw.requires_grad_()
    assert torch.autograd.gradcheck(
        normalize,
        (raw,),
        check_batched_grad=True,
        check_forward_ad=True,
        check_batched_forward_grad=True,
    )
    assert torch.autograd.gradgradcheck(normalize, (raw,))


def _small_layer(kind, transform, dtype=torch.float64):
    """A layer whose second row lies below machine epsilon."""
    layer = _layer(kind, transform, 2, dtype)
    with torch.no_grad():
        offsets = -0.4 * torch.arange(layer.pre_w.shape[-1], dtype=dtype)
        if transform == "identity":
            layer.pre_w[SMALL_ROW] = 1e-20 * torch.exp(offsets)
        else:
            layer.pre_w[SMALL_ROW] = -60.0 + offsets
    return layer


@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("transform", ["exp", "softplus", "identity"])
@pytest.mark.parametrize("kind", KINDS)
def test_forward_mode_transforms_match_reverse_mode(kind, transform, small):
    layer = (
        _small_layer(kind, transform)
        if small
        else _layer(kind, transform, 2, torch.float64)
    )
    x = torch.randn(3, 12, dtype=torch.float64)

    def loss(pre_w):
        output = torch.func.functional_call(layer, {"pre_w": pre_w}, (x,))
        return output.square().sum()

    pre_w = layer.pre_w.detach()
    direction = torch.randn_like(pre_w)
    gradient = torch.func.grad(loss)(pre_w)
    _, directional = torch.func.jvp(loss, (pre_w,), (direction,))
    torch.testing.assert_close(directional, (gradient * direction).sum())
    torch.testing.assert_close(torch.func.jacfwd(loss)(pre_w), gradient)
    # Reverse-over-reverse is no reference: PyTorch's second reverse derivative
    # of a masked torch.norm row is NaN at its zeros, before and after this change.
    torch.testing.assert_close(
        torch.func.hessian(loss)(pre_w),
        torch.func.jacfwd(torch.func.jacfwd(loss))(pre_w),
    )


def test_dual_numbers_match_the_double_backward_jvp():
    pre_w = torch.tensor([[-60.0, -61.0, -62.5], [0.5, -0.2, 0.1]], dtype=torch.float64)
    tangent = torch.tensor([[0.3, -1.0, 2.0], [1.0, 0.5, -0.7]], dtype=torch.float64)

    def normalize(raw):
        return normalize_weight_rows(
            torch.exp(raw), 2, GAMMA, pre_weight=raw, transform="exp"
        )

    with torch.autograd.forward_ad.dual_level():
        dual = normalize(torch.autograd.forward_ad.make_dual(pre_w, tangent))
        actual = torch.autograd.forward_ad.unpack_dual(dual).tangent
    expected = torch.autograd.functional.jvp(normalize, pre_w, tangent)[1]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("transform", ["exp", "softplus"])
def test_rows_just_below_machine_epsilon_are_accurate_to_the_dtype(transform, dtype):
    # The largest weight is eps / 2, where softplus departs most from exp.
    eps = torch.finfo(dtype).eps
    shares = eps * torch.tensor([0.5, 0.25, 0.125, 0.0625], dtype=torch.float64)
    raw = shares.log() if transform == "exp" else torch.log(torch.expm1(shares))
    layer = _layer("indexed", transform, 1, dtype)
    with torch.no_grad():
        layer.pre_w.copy_(raw.to(dtype).expand_as(layer.pre_w))
    assert (layer.sparse_weight().double().sum(-1) < eps).all()
    weights = layer._normalized_sparse_weight().double()
    pre_w = layer.pre_w.detach().double()
    exact = torch.exp(pre_w) if transform == "exp" else F.softplus(pre_w)
    exact = GAMMA * exact / exact.sum(-1, keepdim=True)
    torch.testing.assert_close(weights, exact, rtol=2 * eps, atol=0)


@pytest.mark.parametrize("transform", ["softplus", "identity"])
@pytest.mark.parametrize("kind", KINDS)
def test_compiled_layers_match_eager_forward_and_backward(kind, transform):
    layer = _layer(kind, transform, 2, torch.float32)
    _fill_rows(layer, transform)
    x = torch.randn(3, 12)
    expected = layer(x)  # the first forward resolves (and logs) the backend
    (expected_grad,) = torch.autograd.grad(expected.square().sum(), layer.pre_w)
    torch._dynamo.reset()
    # aot_eager traces forward and backward as single graphs without a compiler.
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
    actual = compiled(x)
    (actual_grad,) = torch.autograd.grad(actual.square().sum(), layer.pre_w)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_grad, expected_grad)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("kind", KINDS)
def test_small_rows_on_cuda_match_the_cpu_reference(kind):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    layer = _layer(kind, "softplus", 2, torch.float32)
    with torch.no_grad():
        # Distinct values, so both devices select the same contacts.
        steps = torch.arange(layer.pre_w.numel(), dtype=torch.float32)
        layer.pre_w.copy_(-90.0 - 0.1 * steps.reshape(layer.pre_w.shape))
    expected = layer.pruned_weight()
    actual = layer.cuda().pruned_weight()
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=0)
    (gradient,) = torch.autograd.grad(actual.square().sum(), layer.pre_w)
    assert torch.isfinite(gradient).all()


@pytest.mark.requires_gpu
@pytest.mark.parametrize("kind", KINDS)
def test_inductor_on_cuda_matches_eager_for_small_rows(kind):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    layer = _layer(kind, "softplus", 2, torch.float32).cuda()
    _fill_rows(layer, "softplus")  # a subnormal row and an underflowed row
    x = torch.randn(16, 12, device="cuda")
    expected = layer(x)
    (expected_grad,) = torch.autograd.grad(expected.square().sum(), layer.pre_w)
    torch._dynamo.reset()
    compiled = torch.compile(layer, fullgraph=True)
    actual = compiled(x)
    (actual_grad,) = torch.autograd.grad(actual.square().sum(), layer.pre_w)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)
