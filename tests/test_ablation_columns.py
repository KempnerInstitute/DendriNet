"""Tests for canonical ablation column naming, flattening, and parsing."""

import pytest

from dendritic_modeling.scripts.sweeps.utils.ablation_columns import (
    discover_ablation_dimensions,
    discover_ablation_module_dimensions,
    flatten_ablation_module_results,
    flatten_ablation_results,
    normalize_metric_key,
    parse_ablation_column,
    parse_ablation_module_column,
    sanitize_module_key,
)


# ---------------------------------------------------------------------------
# normalize_metric_key
# ---------------------------------------------------------------------------
class TestNormalizeMetricKey:
    def test_space_to_underscore(self):
        assert normalize_metric_key("accuracy drop") == "accuracy_drop"

    def test_already_canonical(self):
        assert normalize_metric_key("accuracy_drop") == "accuracy_drop"

    def test_adds_drop_suffix(self):
        assert normalize_metric_key("accuracy") == "accuracy_drop"

    def test_multi_word_metric(self):
        assert (
            normalize_metric_key("categorical_loglikelihood drop")
            == "categorical_loglikelihood_drop"
        )

    def test_pred_label_mi(self):
        assert normalize_metric_key("pred_label_mi_bits") == "pred_label_mi_bits_drop"


class TestSanitizeModuleKey:
    def test_module_slug(self):
        assert sanitize_module_key("core.layers.0/branch") == "core_layers_0_branch"


# ---------------------------------------------------------------------------
# flatten_ablation_results
# ---------------------------------------------------------------------------
class TestFlattenAblationResults:
    @pytest.fixture
    def nested_results(self):
        return {
            "lesion": {
                "excitation": {
                    "layer_0": {
                        "depth": 0,
                        "accuracy_drop": 0.05,
                        "auc_drop": 0.03,
                    },
                    "layer_1": {
                        "depth": 1,
                        "accuracy_drop": 0.10,
                        "categorical_loglikelihood_drop": 0.02,
                    },
                },
                "inhibition": {
                    "layer_0": {
                        "depth": 0,
                        "accuracy_drop": 0.01,
                    },
                },
            },
            "mean_clamp": {
                "upstream": {
                    "layer_2": {
                        "depth": 2,
                        "pred_label_mi_bits_drop": 0.15,
                    },
                },
            },
        }

    def test_basic_flattening(self, nested_results):
        flat = flatten_ablation_results(nested_results)
        assert "ablation_lesion_excitation_accuracy_drop_depth0" in flat
        assert flat["ablation_lesion_excitation_accuracy_drop_depth0"] == 0.05

    def test_multi_word_method(self, nested_results):
        flat = flatten_ablation_results(nested_results)
        assert "ablation_mean_clamp_upstream_pred_label_mi_bits_drop_depth2" in flat
        assert (
            flat["ablation_mean_clamp_upstream_pred_label_mi_bits_drop_depth2"] == 0.15
        )

    def test_multi_word_metric(self, nested_results):
        flat = flatten_ablation_results(nested_results)
        assert (
            "ablation_lesion_excitation_categorical_loglikelihood_drop_depth1" in flat
        )

    def test_skips_metadata_keys(self, nested_results):
        flat = flatten_ablation_results(nested_results)
        # depth, ablation_method should not appear as metric columns
        assert not any("_depth_depth" in k for k in flat)

    def test_multiple_depths(self, nested_results):
        flat = flatten_ablation_results(nested_results)
        assert "ablation_lesion_excitation_accuracy_drop_depth0" in flat
        assert "ablation_lesion_excitation_accuracy_drop_depth1" in flat

    def test_duplicate_depths_are_averaged(self):
        nested = {
            "lesion": {
                "excitation": {
                    "mod_a": {"depth": 1, "accuracy_drop": 0.1},
                    "mod_b": {"depth": 1, "accuracy_drop": 0.3},
                }
            }
        }
        flat = flatten_ablation_results(nested)
        assert flat["ablation_lesion_excitation_accuracy_drop_depth1"] == pytest.approx(
            0.2
        )


class TestFlattenAblationModuleResults:
    def test_preserves_module_identity(self):
        nested = {
            "lesion": {
                "excitation": {
                    "core.layers.0": {"depth": 0, "accuracy_drop": 0.1},
                    "core.layers.1": {"depth": 1, "accuracy_drop": 0.2},
                }
            }
        }
        flat = flatten_ablation_module_results(nested)
        assert (
            "ablation_module__lesion__excitation__accuracy_drop__core_layers_0__depth0"
            in flat
        )
        assert (
            "ablation_module__lesion__excitation__accuracy_drop__core_layers_1__depth1"
            in flat
        )


# ---------------------------------------------------------------------------
# parse_ablation_column
# ---------------------------------------------------------------------------
class TestParseAblationColumn:
    def test_simple_column(self):
        parsed = parse_ablation_column(
            "ablation_lesion_excitation_accuracy_drop_depth0_mean"
        )
        assert parsed is not None
        assert parsed.method == "lesion"
        assert parsed.target == "excitation"
        assert parsed.metric == "accuracy_drop"
        assert parsed.depth == 0
        assert parsed.stat == "mean"

    def test_multi_word_method(self):
        parsed = parse_ablation_column(
            "ablation_mean_clamp_upstream_pred_label_mi_bits_drop_depth2_std"
        )
        assert parsed is not None
        assert parsed.method == "mean_clamp"
        assert parsed.target == "upstream"
        assert parsed.metric == "pred_label_mi_bits_drop"
        assert parsed.depth == 2
        assert parsed.stat == "std"

    def test_categorical_loglikelihood(self):
        parsed = parse_ablation_column(
            "ablation_lesion_all_synapses_categorical_loglikelihood_drop_depth1_mean"
        )
        assert parsed is not None
        assert parsed.method == "lesion"
        assert parsed.target == "all_synapses"
        assert parsed.metric == "categorical_loglikelihood_drop"
        assert parsed.depth == 1

    def test_cosine_similarity(self):
        parsed = parse_ablation_column(
            "ablation_shuffle_inhibition_cosine_similarity_drop_depth0_mean"
        )
        assert parsed is not None
        assert parsed.method == "shuffle"
        assert parsed.target == "inhibition"
        assert parsed.metric == "cosine_similarity_drop"

    def test_no_stat_suffix(self):
        parsed = parse_ablation_column(
            "ablation_lesion_excitation_accuracy_drop_depth0"
        )
        assert parsed is not None
        assert parsed.stat is None

    def test_non_ablation_column_returns_none(self):
        assert parse_ablation_column("accuracy_mean") is None
        assert parse_ablation_column("performance_accuracy") is None

    def test_base_col_property(self):
        parsed = parse_ablation_column(
            "ablation_lesion_excitation_accuracy_drop_depth0_mean"
        )
        assert parsed.base_col == "ablation_lesion_excitation_accuracy_drop_depth0"

    def test_full_col_property(self):
        parsed = parse_ablation_column(
            "ablation_lesion_excitation_accuracy_drop_depth0_mean"
        )
        assert parsed.full_col == "ablation_lesion_excitation_accuracy_drop_depth0_mean"


class TestParseAblationModuleColumn:
    def test_module_column(self):
        parsed = parse_ablation_module_column(
            "ablation_module__mean_clamp__upstream__pred_label_mi_bits_drop__core_layers_1__depth2_mean"
        )
        assert parsed is not None
        assert parsed.method == "mean_clamp"
        assert parsed.target == "upstream"
        assert parsed.metric == "pred_label_mi_bits_drop"
        assert parsed.module == "core_layers_1"
        assert parsed.depth == 2
        assert parsed.stat == "mean"


# ---------------------------------------------------------------------------
# discover_ablation_dimensions
# ---------------------------------------------------------------------------
class TestDiscoverAblationDimensions:
    def test_discovers_all_dimensions(self):
        cols = [
            "ablation_lesion_excitation_accuracy_drop_depth0_mean",
            "ablation_lesion_excitation_accuracy_drop_depth0_std",
            "ablation_lesion_inhibition_accuracy_drop_depth1_mean",
            "ablation_mean_clamp_upstream_pred_label_mi_bits_drop_depth2_mean",
            "not_an_ablation_col",
        ]
        methods, targets, metrics, depths = discover_ablation_dimensions(cols)
        assert methods == ["lesion", "mean_clamp"]
        assert targets == ["excitation", "inhibition", "upstream"]
        assert metrics == ["accuracy_drop", "pred_label_mi_bits_drop"]
        assert depths == [0, 1, 2]

    def test_empty_columns(self):
        methods, targets, metrics, depths = discover_ablation_dimensions([])
        assert methods == []
        assert targets == []
        assert metrics == []
        assert depths == []


class TestDiscoverAblationModuleDimensions:
    def test_discovers_module_dimensions(self):
        cols = [
            "ablation_module__lesion__excitation__accuracy_drop__core_layers_0__depth0_mean",
            "ablation_module__mean_clamp__upstream__pred_label_mi_bits_drop__core_layers_1__depth2_mean",
        ]
        methods, targets, metrics, modules, depths = (
            discover_ablation_module_dimensions(cols)
        )
        assert methods == ["lesion", "mean_clamp"]
        assert targets == ["excitation", "upstream"]
        assert metrics == ["accuracy_drop", "pred_label_mi_bits_drop"]
        assert modules == ["core_layers_0", "core_layers_1"]
        assert depths == [0, 2]


# ---------------------------------------------------------------------------
# Round-trip: flatten -> parse
# ---------------------------------------------------------------------------
class TestRoundTrip:
    def test_flatten_then_parse(self):
        """Flatten nested results, then parse the resulting column names."""
        nested = {
            "lesion": {
                "excitation": {
                    "mod_a": {"depth": 0, "accuracy_drop": 0.1},
                },
            },
            "mean_clamp": {
                "upstream": {
                    "mod_b": {"depth": 2, "pred_label_mi_bits_drop": 0.5},
                },
            },
        }
        flat = flatten_ablation_results(nested)

        # Every flat column should be parseable
        for col in flat:
            parsed = parse_ablation_column(col)
            assert parsed is not None, f"Failed to parse: {col}"

        # Check specific round-trip
        parsed = parse_ablation_column(
            "ablation_mean_clamp_upstream_pred_label_mi_bits_drop_depth2"
        )
        assert parsed.method == "mean_clamp"
        assert parsed.target == "upstream"
        assert parsed.metric == "pred_label_mi_bits_drop"
        assert parsed.depth == 2
