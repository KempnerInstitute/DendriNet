"""Smoke tests: every example config builds + forwards finite.

Closes the coverage gap where ~half of configs/examples/*.yaml were never
smoke-tested. Conv-backbone (alexnet/pretrained/fsdp) and transformer-replacement
configs need image/sequence inputs or a teacher and are exercised elsewhere, so
they are skipped here with an explicit reason rather than silently omitted.
"""

from __future__ import annotations

import glob
import os

import pytest
import torch

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

try:
    from dendritic_modeling.config.recurrence import is_recurrent_core_config
except Exception:  # pragma: no cover
    is_recurrent_core_config = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_CONFIGS = sorted(glob.glob(os.path.join(REPO_ROOT, "configs/examples/*.yaml")))
TRANSFORMER_REPLACEMENT_CONFIGS = sorted(
    glob.glob(
        os.path.join(REPO_ROOT, "configs/examples/transformer_replacement/*.yaml")
    )
)

# Need a conv image tensor / pretrained backbone / teacher model — covered by
# dedicated tests, not by this flat-input smoke harness.
SKIP_SUBSTRINGS = (
    "alexnet",
    "pretrained",
    "fsdp",
    "transformer_replacement",
    "deepseek",
)


def _needs_special_input(name: str) -> bool:
    return any(s in name for s in SKIP_SUBSTRINGS)


def _infer_input_dim(model, cfg) -> int:
    enc = getattr(model, "encoder", None)
    for attr in ("output_dim", "input_dim"):
        v = getattr(enc, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    ip = cfg.model.encoder.params.input_dim
    return ip if isinstance(ip, int) and ip > 0 else 784


@pytest.mark.parametrize(
    "config_path", ALL_CONFIGS, ids=[os.path.basename(p) for p in ALL_CONFIGS]
)
def test_example_config_builds_and_forwards_finite(config_path):
    name = os.path.basename(config_path)
    if _needs_special_input(name):
        pytest.skip(
            f"{name} needs conv/pretrained/transformer input — covered elsewhere"
        )

    load_config.cache_clear()
    cfg = load_config(config_path)
    tr = getattr(cfg.model, "transformer_replacement", None)
    if tr is not None and getattr(tr, "enabled", False):
        pytest.skip(f"{name} is a transformer-replacement config")

    torch.manual_seed(0)
    model, _ = initialize_model(cfg.model)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    assert n > 0

    recurrent = False
    if is_recurrent_core_config is not None:
        try:
            recurrent = bool(is_recurrent_core_config(cfg.model.core))
        except Exception:
            recurrent = False

    dim = _infer_input_dim(model, cfg)
    x = torch.rand(2, 4, dim) if recurrent else torch.rand(2, dim)
    out = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.isfinite(logits).all(), f"{name}: non-finite forward output"


@pytest.mark.parametrize(
    "config_path",
    TRANSFORMER_REPLACEMENT_CONFIGS,
    ids=[os.path.basename(path) for path in TRANSFORMER_REPLACEMENT_CONFIGS],
)
def test_transformer_replacement_example_config_is_consistent(config_path):
    load_config.cache_clear()
    cfg = load_config(config_path)
    model_cfg = cfg.model.transformer_replacement
    train_cfg = cfg.training.transformer_replacement

    assert model_cfg.enabled and train_cfg.enabled
    assert model_cfg.layers == train_cfg.layers
    assert len(model_cfg.layers) == len(set(model_cfg.layers))
    assert all(layer >= 0 for layer in model_cfg.layers)
    assert model_cfg.model_name == "allenai/OLMo-2-0425-1B"
    assert model_cfg.replacement_kwargs["kind"] == "gated_dendritic_ffn"
    expected_teacher_init = train_cfg.student_initialization != "from_config"
    assert model_cfg.replacement_kwargs["teacher_topk_init"] is expected_teacher_init
    assert model_cfg.replacement_kwargs["output_topk"] == 6144

    population = cfg.model.core.population_network["layers"][0]
    assert population["populations"][0]["n_neurons"] == 8192
    assert population["populations"][0]["branch_factors"] == []
    assert population["population_defaults"]["ff_excitatory_synapses"] == 1536


@pytest.mark.parametrize(
    "core_type,expect_shunting",
    [
        ("rnn_dendritic_shunting", True),
        ("rnn_dendritic_additive", False),
        ("rnn_flat_shunting", True),
        ("rnn_flat_additive", False),
    ],
)
def test_recurrent_unified_ei_aliases_build(core_type, expect_shunting):
    """Covers the rnn_dendritic_*/rnn_flat_* core.type aliases. rnn_dendritic_shunting
    is the config_exp.yaml default and previously had zero test coverage."""
    import warnings

    from omegaconf import OmegaConf

    from dendritic_modeling.networks.architectures.factory import get_architecture

    branch_factors = [1] if "flat" in core_type else [2]
    params = OmegaConf.create(
        {
            "unified_ei": {
                "output_mode": "last",
                "layers": [
                    {
                        "excitatory": {
                            "n_neurons": 6,
                            "branch_factors": branch_factors,
                            "ff_excitatory_synapses": 3,
                            "rec_excitatory_synapses": 2,
                        },
                        "inhibitory": None,
                        "recurrent": True,
                        "dt": 1.0,
                    }
                ],
            }
        }
    )
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = get_architecture(core_type, params, input_dim=8)
    layer = net.layers[0].e_population.branch_layers[0]
    assert layer.use_shunting is expect_shunting
    out = net(torch.rand(2, 4, 8))
    logits = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.isfinite(logits).all(), f"{core_type}: non-finite forward"
