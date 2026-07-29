import json
from pathlib import Path

import pytest

from dendritic_modeling.config import load_config
from dendritic_modeling.networks.architectures.transformer import (
    DendriticFFNReplacement,
    GatedDendriticFFNReplacement,
    TransformerParameterSpec,
    estimate_dendritic_ffn_params,
    estimate_factorized_lm_head_params,
    estimate_factorized_token_embedding_params,
    estimate_indexed_attention_params,
    profile_dendritic_lm,
    profile_dendritic_lm_config,
    profile_transformer_component_options,
)
from dendritic_modeling.scripts.transformer_replacement._olmo_generators import (
    generate_all_layer_configs,
    generate_joint_configs,
)
from dendritic_modeling.scripts.transformer_replacement.assemble_layer_checkpoints import (
    assemble_layer_checkpoints,
)

OLMO3_7B_CONFIG = {
    "model_type": "olmo3",
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "vocab_size": 100278,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "rms_norm_eps": 1e-6,
}


def test_olmo3_parameter_spec_matches_checkpoint_metadata_count():
    spec = TransformerParameterSpec.from_hf_config(OLMO3_7B_CONFIG)
    breakdown = spec.parameter_breakdown()

    assert spec.head_dim == 128
    assert spec.attention_norms_per_layer == 2
    assert breakdown["total"] == 7_298_011_136
    assert breakdown["dense_ffn_per_layer"] == 135_266_304


def test_full_density_indexed_attention_matches_dense_projection_count():
    spec = TransformerParameterSpec.from_hf_config(OLMO3_7B_CONFIG)

    estimate = estimate_indexed_attention_params(
        spec,
        query_topk=spec.hidden_size,
        key_value_topk=spec.hidden_size,
        output_topk=spec.num_attention_heads * spec.head_dim,
    )

    assert estimate["stored_total"] == spec.parameter_breakdown()["attention_per_layer"]
    assert estimate["stored_reduction_factor"] == pytest.approx(1.0)
    assert estimate["row_bitmask_topology_bytes"] == 4 * 4096 * (4096 // 8)


def test_indexed_attention_validates_each_projection_input_width():
    spec = TransformerParameterSpec.from_hf_config(OLMO3_7B_CONFIG)

    with pytest.raises(ValueError, match="output_topk"):
        estimate_indexed_attention_params(
            spec,
            query_topk=1024,
            key_value_topk=1024,
            output_topk=spec.hidden_size + 1,
        )


def test_indexed_attention_full_density_matches_grouped_query_bias_count():
    config = {
        **OLMO3_7B_CONFIG,
        "num_key_value_heads": 8,
        "attention_bias": True,
    }
    spec = TransformerParameterSpec.from_hf_config(config)

    estimate = estimate_indexed_attention_params(
        spec,
        query_topk=spec.hidden_size,
        key_value_topk=spec.hidden_size,
        output_topk=spec.hidden_size,
    )

    assert estimate["stored_total"] == spec.parameter_breakdown()["attention_per_layer"]


def test_factorized_embedding_and_head_report_storage_and_access_costs():
    spec = TransformerParameterSpec.from_hf_config(OLMO3_7B_CONFIG)

    embedding = estimate_factorized_token_embedding_params(spec, rank=512)
    head = estimate_factorized_lm_head_params(spec, rank=512)

    assert embedding["stored_total"] == 512 * (spec.vocab_size + spec.hidden_size)
    assert embedding["stored_total"] < embedding["dense_total"]
    assert (
        embedding["factorized_weights_touched_per_token"]
        > embedding["dense_weights_touched_per_token"]
    )
    assert head["stored_total"] == embedding["stored_total"]
    assert head["independent_replacement_applicable"]

    with pytest.raises(ValueError, match="rank"):
        estimate_factorized_token_embedding_params(spec, rank=0)


def test_component_profile_combines_olmo_attention_embedding_and_head_options():
    spec = TransformerParameterSpec.from_hf_config(OLMO3_7B_CONFIG)

    report = profile_transformer_component_options(
        spec,
        attention_layers=range(spec.num_hidden_layers),
        query_topk=3072,
        key_value_topk=3072,
        output_topk=3072,
        embedding_rank=512,
        lm_head_rank=512,
    )

    assert report["options"]["attention"]["num_layers"] == 32
    assert report["options"]["attention"]["per_layer"][
        "stored_reduction_factor"
    ] == pytest.approx(4 / 3)
    assert (
        report["options"]["attention"]["per_layer"]["runtime_int64_topology_bytes"]
        > 2 * report["options"]["attention"]["per_layer"]["stored_total"]
    )
    assert report["combined_candidate"]["applicable"]
    assert (
        report["combined_candidate"]["stored_parameters"] < report["baseline"]["total"]
    )


def test_component_profile_rejects_independent_head_for_tied_baseline():
    tied_config = {**OLMO3_7B_CONFIG, "tie_word_embeddings": True}
    spec = TransformerParameterSpec.from_hf_config(tied_config)

    report = profile_transformer_component_options(spec, lm_head_rank=256)

    assert not report["options"]["lm_head"]["independent_replacement_applicable"]
    assert report["options"]["lm_head"]["student_stored_parameters"] is None
    assert not report["combined_candidate"]["applicable"]


def test_component_profiles_reject_ambiguous_attention_requests():
    spec = TransformerParameterSpec.from_hf_config(OLMO3_7B_CONFIG)

    with pytest.raises(ValueError, match="set together"):
        profile_transformer_component_options(
            spec,
            attention_layers=[0],
            query_topk=1024,
        )
    with pytest.raises(ValueError, match="duplicates"):
        profile_transformer_component_options(
            spec,
            attention_layers=[0, 0],
            query_topk=1024,
            key_value_topk=1024,
            output_topk=1024,
        )
    with pytest.raises(ValueError, match="duplicates"):
        profile_dendritic_lm(
            spec,
            dendritic_per_layer={"stored_total": 1, "active_total": 1},
            replaced_layers=[0, 0],
        )


def test_dendritic_ffn_parameter_estimate_matches_real_module():
    module = DendriticFFNReplacement(
        hidden_size=8,
        dendritic_units=4,
        branch_factors=[2, 2],
        synapses_per_branch=3,
        topk_type="indexed_rewire",
        use_shunting=False,
        pre_norm=True,
    )

    estimate = module.parameter_estimate()

    assert estimate["stored_total"] == sum(
        parameter.numel() for parameter in module.parameters()
    )
    assert estimate["reactivation"] == 56
    assert estimate["pre_norm"] == 16


def test_gated_dendritic_ffn_parameter_estimate_matches_real_module():
    module = GatedDendriticFFNReplacement(
        hidden_size=8,
        dendritic_units=4,
        branch_factors=[2, 2],
        synapses_per_branch=3,
        topk_type="indexed_rewire",
        use_shunting=False,
        pre_norm=True,
    )

    estimate = module.parameter_estimate()

    assert estimate["stored_total"] == sum(
        parameter.numel() for parameter in module.parameters()
    )
    assert estimate["reactivation"] == 112
    assert estimate["pre_norm"] == 16


def test_gated_sparse_output_parameter_estimate_matches_real_module():
    module = GatedDendriticFFNReplacement(
        hidden_size=8,
        dendritic_units=16,
        branch_factors=[],
        synapses_per_branch=4,
        input_transform="identity",
        topk_type="indexed",
        use_shunting=False,
        reactivate=False,
        weight_transform="identity",
        output_topk=5,
    )

    estimate = module.parameter_estimate()

    assert estimate["output_projection"] == 8 * 5
    assert estimate["stored_total"] == sum(
        parameter.numel() for parameter in module.parameters()
    )


def test_scaling_profile_accounts_for_gated_replacement_kind():
    config_path = Path(
        "configs/transformer_replacement_benchmarks/olmo2_7b_dolmino_layer0_2048.yaml"
    )
    load_config.cache_clear()
    config = load_config(str(config_path))
    config.model.transformer_replacement.replacement_kwargs["kind"] = (
        "gated_dendritic_ffn"
    )
    config.model.transformer_replacement.replacement_kwargs["output_topk"] = 1024

    report = profile_dendritic_lm_config(config, OLMO3_7B_CONFIG)

    per_layer = report["replacement"]["per_layer"]
    assert report["replacement"]["architecture"]["replacement_kind"] == (
        "gated_dendritic_ffn"
    )
    assert per_layer["stored_synapses"] == 2 * 917_504
    assert per_layer["output_projection"] == 4_194_304


@pytest.mark.parametrize(
    ("topk_type", "extra"),
    [
        ("standard", {}),
        ("dense_to_sparse", {}),
        ("indexed", {}),
        ("indexed_rewire", {}),
        ("indexed_dynamic", {"indexed_candidate_size": 6}),
    ],
)
def test_parameter_estimate_matches_all_topk_storage_modes(topk_type, extra):
    module = DendriticFFNReplacement(
        hidden_size=8,
        dendritic_units=4,
        branch_factors=[2],
        synapses_per_branch=3,
        topk_type=topk_type,
        use_shunting=False,
        **extra,
    )

    estimate = module.parameter_estimate()

    assert estimate["stored_total"] == sum(
        parameter.numel() for parameter in module.parameters()
    )
    assert estimate["active_total"] <= estimate["stored_total"]


def test_profile_distinguishes_stored_parameters_from_active_proxy():
    spec = TransformerParameterSpec.from_hf_config(OLMO3_7B_CONFIG)
    dendritic = estimate_dendritic_ffn_params(
        hidden_size=spec.hidden_size,
        dendritic_units=512,
        branch_factors=[2, 2],
        synapses_per_branch=32,
        candidate_size=256,
        topk_type="indexed_dynamic",
    )

    report = profile_dendritic_lm(
        spec,
        dendritic_per_layer=dendritic,
        replaced_layers=range(spec.num_hidden_layers),
    )

    assert (
        report["student"]["stored_parameters"]
        > report["student"]["active_weight_count_proxy"]
    )
    assert (
        report["student"]["ffn_active_reduction_factor"]
        > report["student"]["ffn_stored_reduction_factor"]
    )
    assert "not a FLOP" in report["definitions"]["active_weight_count_proxy"]


def test_profile_reads_existing_replacement_yaml_without_model_weights():
    config_path = Path(
        "configs/transformer_replacement_benchmarks/olmo2_7b_dolmino_layer0_2048.yaml"
    )
    load_config.cache_clear()
    config = load_config(str(config_path))

    report = profile_dendritic_lm_config(config, OLMO3_7B_CONFIG)

    assert report["replacement"]["layers"] == [0]
    assert report["replacement"]["architecture"]["dendritic_units"] == 2048
    assert report["replacement"]["per_layer"]["stored_total"] > 0


def test_olmo_generator_writes_13b_layer_manifest(tmp_path):
    config_root = tmp_path / "configs"
    results_root = tmp_path / "results"

    manifest = generate_all_layer_configs(
        variant="13b",
        config_root=config_root,
        results_root=results_root,
        max_steps=3,
        num_layers=2,
    )

    assert manifest["model_name"] == "allenai/OLMo-2-1124-13B"
    assert len(manifest["layer_runs"]) == 2
    assert Path(manifest["layer_configs"][0]).is_file()
    saved_manifest = json.loads(
        (Path(manifest["config_dir"]) / "manifest.json").read_text()
    )
    assert saved_manifest["assembled_checkpoint_dir"].endswith("all_layer_checkpoints")


def test_joint_generator_uses_full_depth_hidden_supervision(tmp_path):
    manifest = generate_joint_configs(
        variant="13b",
        config_root=tmp_path / "configs",
        results_root=tmp_path / "results",
        max_steps=3,
    )
    payload = load_config(manifest["configs"][0]["config"])

    assert payload.training.transformer_replacement.hidden_loss_layers == [9, 19, 39]


def test_assemble_layer_checkpoints_creates_flat_warm_start(tmp_path):
    results = tmp_path / "results"
    runs = []
    for layer in range(2):
        checkpoint = results / f"layer_{layer:02d}" / f"layer_{layer}_replacement.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"layer-{layer}".encode())
        runs.append({"layer": layer, "checkpoint": str(checkpoint)})
    destination = results / "all_layer_checkpoints"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "num_layers": 2,
                "layer_runs": runs,
                "assembled_checkpoint_dir": str(destination),
            }
        )
    )

    result = assemble_layer_checkpoints(manifest_path)

    assert result["num_layers"] == 2
    assert (destination / "layer_0_replacement.pt").read_bytes() == b"layer-0"
    assert (destination / "layer_1_replacement.pt").is_symlink()
    assert (destination / "assembly_manifest.json").is_file()
    with pytest.raises(FileExistsError, match="partial assembly"):
        assemble_layer_checkpoints(manifest_path)
