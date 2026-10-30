"""Numerical, checkpoint and configuration contracts of feature readouts."""

from dataclasses import asdict, replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dendritic_modeling.config import (
    AffineMLPConfig,
    CoreConfig,
    FeatureReadoutConfig,
    config_from_mapping,
)
from dendritic_modeling.config.seeds import derive_seed
from dendritic_modeling.networks.architectures.classical.feature_readout import (
    build_feature_readout,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model


def spec(normalization="none", hidden=False):
    return FeatureReadoutConfig(
        input_shape=[8, 2, 2],
        readout=AffineMLPConfig(
            dimensions=[32, 7, 3] if hidden else [32, 3],
            initialization="pytorch_default",
            activate_output=False,
        ),
        normalization=normalization,
        groups=2 if normalization in {"group_mean", "group_norm"} else None,
        divisor_offset=0.05 if normalization == "group_mean" else None,
    )


class Reference(nn.Module):
    """Independent literal Torch construction, not the tested affine builder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        if len(config.readout.dimensions) == 3:
            self.readout = nn.Sequential(nn.Linear(32, 7), nn.ReLU(), nn.Linear(7, 3))
        else:
            self.readout = nn.Linear(32, 3)
        if config.normalization == "layer_norm":
            self.normalizer = nn.LayerNorm((8, 2, 2), elementwise_affine=True)
        elif config.normalization == "group_norm":
            self.normalizer = nn.GroupNorm(2, 8, affine=True)
        else:
            self.normalizer = None

    def forward(self, x, *, divisor_noise=None):
        if self.config.normalization == "group_mean":
            grouped = x.reshape(x.shape[0], 2, 4, 2, 2)
            divisor = grouped.mean(dim=(2, 3, 4), keepdim=True)
            if divisor_noise is not None:
                divisor = divisor * divisor_noise[:, :, None, None, None]
            x = (grouped / (0.05 + divisor.clamp_min(0.0))).reshape_as(x)
        elif self.normalizer is not None:
            x = self.normalizer(x)
        return self.readout(x.flatten(1))


@pytest.mark.parametrize(
    "normalization", ["none", "group_mean", "layer_norm", "group_norm"]
)
@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
)
def test_initialization_gradients_updates_and_checkpoint_exact(
    normalization, hidden, dtype, device
):
    config = spec(normalization, hidden)
    torch.manual_seed(23)
    old = Reference(config).to(device=device, dtype=dtype)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(23)
    new = build_feature_readout(config).to(device=device, dtype=dtype)
    assert torch.equal(rng, torch.get_rng_state())
    assert list(old.state_dict()) == list(new.state_dict())
    assert list(dict(old.named_parameters())) == list(dict(new.named_parameters()))
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    x = torch.linspace(-0.2, 2.4, 96, dtype=dtype, device=device).reshape(3, 8, 2, 2)
    noise = (
        torch.linspace(0.2, 2, 6, dtype=dtype, device=device).reshape(3, 2)
        if normalization == "group_mean"
        else None
    )
    histories = []
    for model in (old, new):
        history = []
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            xx = x.clone().requires_grad_(True)
            output = model(xx, divisor_noise=noise)
            output.square().mean().backward()
            history.extend([output.detach().clone(), xx.grad.clone()])
            history.extend(p.grad.clone() for p in model.parameters())
            optimizer.step()
            history.extend(p.detach().clone() for p in model.parameters())
        histories.append(history)
    assert all(torch.equal(a, b) for a, b in zip(*histories))
    new.load_state_dict(old.state_dict(), strict=True)
    assert torch.equal(new(x, divisor_noise=noise), old(x, divisor_noise=noise))
    assert torch.equal(
        new(x.flatten(1), divisor_noise=noise), old(x, divisor_noise=noise)
    )


def mapping():
    return {
        "config_version": 4,
        "model": {
            "task": "classification",
            "output_scale_mode": "fixed",
            "encoder": {"type": "identity", "params": {"input_dim": 32}},
            "core": {"type": "feature_readout", "feature_readout": asdict(spec())},
            "decoder": {"type": "identity"},
        },
    }


def test_config_factory_and_full_builder_are_exact():
    cfg = config_from_mapping(mapping())
    constructors = [
        lambda: build_feature_readout(cfg.model.core.feature_readout),
        lambda: get_architecture(
            "feature_readout", cfg.model.core, input_dim=32, suffix_input_dim=3
        ),
        lambda: initialize_model(cfg.model)[0].core_network,
    ]
    states, rngs = [], []
    for build in constructors:
        torch.manual_seed(24)
        model = build()
        states.append(model.state_dict())
        rngs.append(torch.get_rng_state().clone())
        assert model.feature_readout_config == asdict(cfg.model.core.feature_readout)
    assert all(torch.equal(rngs[0], r) for r in rngs)
    assert all(torch.equal(states[0][k], s[k]) for s in states for k in s)
    with pytest.raises(ValueError, match="encoder"):
        get_architecture("feature_readout", cfg.model.core, input_dim=31)
    with pytest.raises(ValueError, match="decoder"):
        get_architecture("feature_readout", cfg.model.core, suffix_input_dim=4)
    with pytest.raises(ValueError, match="morphology"):
        get_architecture(
            "feature_readout",
            {
                "feature_readout": asdict(spec()),
                "morphology": {"leak_conductance": 2.0},
            },
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"normalization": "shunting"},
        {"groups": 2},
        {"divisor_offset": 1},
        {"normalization_affine": False},
        {"normalization_epsilon": 0.1},
        {"input_shape": [True, 2]},
        {"input_shape": [8, 2, 3]},
        {"normalization": "group_mean", "groups": 3, "divisor_offset": 0.05},
        {"normalization": "group_mean", "groups": 2, "divisor_offset": 0},
        {"normalization": "group_mean", "groups": 2, "divisor_offset": float("nan")},
        {"normalization": "layer_norm", "normalization_epsilon": float("inf")},
        {
            "readout": AffineMLPConfig(
                dimensions=[32, 3],
                mode="fixed",
                fanin=[4],
                connectivity_seeds=[1],
                activate_output=False,
            )
        },
    ],
)
def test_invalid_and_ignored_options_rejected(changes):
    with pytest.raises(ValueError):
        build_feature_readout(replace(spec(), **changes))


def test_config_rejects_unknown_unused_and_unsupported_options():
    value = mapping()
    value["model"]["core"]["feature_readout"]["readout"]["initializaton"] = "foo"
    with pytest.raises(ValueError, match="initializaton"):
        config_from_mapping(value)
    value = mapping()
    value["model"]["core"]["type"] = "affine_mlp"
    with pytest.raises(ValueError, match="feature_readout"):
        config_from_mapping(value)
    value = mapping()
    value["training"] = {"main": {"strategy": "local_ca"}}
    with pytest.raises(ValueError, match="backpropagation"):
        config_from_mapping(value)
    value = mapping()
    value["model"]["core"]["biological_neuron"] = True
    with pytest.raises(ValueError, match="biological"):
        config_from_mapping(value)


def test_omitted_section_and_explicit_none_leave_existing_core_unchanged():
    assert CoreConfig().feature_readout is None
    base = {"config_version": 4, "model": {"core": {"biological_neuron": False}}}
    old_cfg = config_from_mapping(base)
    base["model"]["core"]["feature_readout"] = None
    new_cfg = config_from_mapping(base)
    torch.manual_seed(25)
    old, _ = initialize_model(old_cfg.model)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(25)
    new, _ = initialize_model(new_cfg.model)
    assert torch.equal(rng, torch.get_rng_state())
    assert list(old.state_dict()) == list(new.state_dict())
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())


def test_explicit_seed_selects_a_keyed_stream():
    config = spec("group_mean")
    rng = torch.get_rng_state().clone()
    raw = {"feature_readout": asdict(config), "initialization_seed": 5}
    a = get_architecture("feature_readout", raw)
    b = get_architecture("feature_readout", raw)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    # Keyed by the core type, so an encoder built after manual_seed(5) is not
    # replayed by the readout.
    built = []
    for seed in (derive_seed(5, "core.feature_readout.initialization"), 5):
        torch.manual_seed(seed)
        built.append(build_feature_readout(config).readout.weight)
    assert torch.equal(a.readout.weight, built[0])
    assert not torch.equal(a.readout.weight, built[1])


def test_direct_construction_copies_nested_config():
    torch.manual_seed(26)
    config = spec("group_mean")
    expected_config = asdict(config)
    direct = build_feature_readout(config)
    x = torch.linspace(-0.2, 2.4, 64).reshape(2, 8, 2, 2)
    expected_output = direct(x).detach().clone()

    config.input_shape[0] = 99
    config.readout.dimensions[0] = 99
    config.divisor_offset = 1000.0

    assert asdict(direct.config) == expected_config
    assert direct.feature_readout_config == expected_config
    torch.testing.assert_close(direct(x), expected_output, rtol=0, atol=0)


def test_training_initializes_the_readout_like_the_affine_mlp_control():
    from dendritic_modeling.scripts.script_utils.setup_utils import (
        _apply_initialization_seed_defaults,
    )

    # The encoder draws from the model stream first; reseeding the readout
    # with the model seed would replay those draws instead of continuing.
    encoder = {
        "type": "mlpautoencoder",
        "params": {
            "input_dim": 32,
            "encoder_hidden_dims": [32],
            "latent_dim": 32,
            "decoder_hidden_dims": [32],
        },
    }
    readout, control = mapping(), mapping()
    control["model"]["core"] = {
        "type": "affine_mlp",
        "affine_mlp": asdict(spec().readout),
    }
    states = []
    for value in (readout, control):
        value["model"]["encoder"] = encoder
        cfg = config_from_mapping(value)
        _apply_initialization_seed_defaults(cfg.model, 41)
        torch.manual_seed(41)  # As training's set_seed(model_seed).
        states.append(list(initialize_model(cfg.model)[0].state_dict().values()))
        if value is readout:
            assert cfg.model.core.initialization_seed is None
    assert len(states[0]) == len(states[1])
    assert all(torch.equal(a, b) for a, b in zip(*states))


def test_group_mean_divisor_clamps_negative_means_at_the_offset():
    model = build_feature_readout(spec("group_mean")).double()
    x = torch.linspace(-2.0, 1.0, 64, dtype=torch.float64).reshape(2, 8, 2, 2)
    groups = x.reshape(2, 2, 4, 2, 2)
    means = groups.mean(dim=(2, 3, 4), keepdim=True)
    assert (means < 0).any() and (means > 0).any()
    divisor = 0.05 + torch.where(means > 0, means, torch.zeros_like(means))
    assert torch.equal(model.representation(x), (groups / divisor).reshape(2, 32))


@pytest.mark.parametrize("normalization", ["layer_norm", "group_norm"])
def test_normalization_epsilon_and_affine_options_reach_the_torch_module(
    normalization,
):
    config = replace(
        spec(normalization), normalization_epsilon=1e-3, normalization_affine=False
    )
    model = build_feature_readout(config).double()
    assert not any(name.startswith("normalizer.") for name in model.state_dict())
    x = torch.linspace(-0.2, 2.4, 96, dtype=torch.float64).reshape(3, 8, 2, 2)
    normalized = (
        F.layer_norm(x, (8, 2, 2), eps=1e-3)
        if normalization == "layer_norm"
        else F.group_norm(x, 2, eps=1e-3)
    )
    assert torch.equal(model(x), model.readout(normalized.flatten(1)))


def test_group_divisor_noise_validation_and_gradient():
    model = build_feature_readout(spec("group_mean")).double()
    x = (
        torch.linspace(0.1, 2, 64, dtype=torch.float64)
        .reshape(2, 8, 2, 2)
        .requires_grad_(True)
    )
    noise = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda xx, nn: model(xx, divisor_noise=nn), (x, noise)
    )
    for invalid in [torch.ones(2), torch.ones(2, 2), noise * float("nan"), -noise]:
        with pytest.raises(ValueError, match="divisor_noise"):
            model(x, divisor_noise=invalid)
    with pytest.raises(ValueError, match="only applies"):
        build_feature_readout(spec())(x.float(), divisor_noise=noise.float())
    with pytest.raises(ValueError, match="input_shape"):
        model(torch.zeros(2, 4, 8, dtype=torch.float64))
    with pytest.raises(TypeError, match="floating"):
        model(torch.zeros(2, 32, dtype=torch.int64))


@pytest.mark.parametrize(
    "dtype, offset",
    [
        (torch.float16, 1e-8),
        (torch.float16, 1e5),
        (torch.bfloat16, 1e-42),
        (torch.bfloat16, 1e40),
        (torch.float32, 1e-50),
        (torch.float32, 1e40),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
)
def test_group_mean_offset_must_remain_positive_and_finite(dtype, offset, device):
    config = replace(spec("group_mean"), divisor_offset=offset)
    model = build_feature_readout(config).to(device=device, dtype=torch.float64)
    x = torch.zeros(2, 32, device=device, dtype=torch.float64)
    assert torch.isfinite(model(x)).all()
    model.to(dtype)
    for call in (model, model.representation):
        with pytest.raises(ValueError, match="divisor_offset"):
            call(x.to(dtype))


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    "normalization", ["none", "group_mean", "layer_norm", "group_norm"]
)
def test_explicit_seed_under_cuda_default_preserves_caller_state(normalization):
    raw = {"feature_readout": asdict(spec(normalization)), "initialization_seed": 7}
    reference = get_architecture("feature_readout", raw)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    with torch.device("cuda"):
        first = get_architecture("feature_readout", raw)
        second = get_architecture("feature_readout", raw)
        assert torch.empty(0).device.type == "cuda"
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        for name, value in reference.state_dict().items():
            for model in (first, second):
                actual = model.state_dict()[name]
                assert actual.device.type == "cpu"
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
