from dendritic_modeling.analysis.core.ablation import (
    ABLATION_VIDEO_METRIC_LABELS,
    _has_metric_drop_nested,
    flatten_3level_ablation,
)


def test_has_metric_drop_nested_matches_nested_dict_and_sequence_rules():
    payload = {
        "lesion_excitation": {
            "layer0": {"depth": 0, "accuracy_drop": 0.1},
            "layer1": [{"mse_drop": 0.2}],
        },
        "metadata": ("not-a-metric", {"auc_drop": 0.3}),
    }

    assert _has_metric_drop_nested(payload, "accuracy")
    assert _has_metric_drop_nested(payload, "mse")
    assert _has_metric_drop_nested(payload, "auc")
    assert not _has_metric_drop_nested(payload, "cosine_similarity")
    assert not _has_metric_drop_nested(1.0, "accuracy")


def test_ablation_video_metric_labels_preserve_supported_order():
    assert list(ABLATION_VIDEO_METRIC_LABELS) == [
        "accuracy",
        "auc",
        "categorical_loglikelihood",
        "mse",
        "cosine_similarity",
        "pred_label_mi_bits",
    ]


def test_flatten_3level_ablation_preserves_legacy_two_level_payloads():
    payload = {
        "lesion": {
            "layer0": {"depth": 0, "accuracy_drop": 0.1},
        }
    }

    assert flatten_3level_ablation(payload) == payload
