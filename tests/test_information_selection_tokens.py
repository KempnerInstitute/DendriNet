"""Information selector tokens are checked when the config loads."""

from __future__ import annotations

import pytest

from dendritic_modeling.config import config_from_mapping


def _information(params: dict):
    return config_from_mapping(
        {"analysis": {"information": {"enabled": True, "params": params}}}
    ).analysis.information.params


def test_known_tokens_and_spellings_load():
    params = _information(
        {
            "selection": {
                "components": ["basic", "Pairwise-MI", "soma_coupling"],
                "scopes": ["per layer"],
                "signal_variants": ["network", "lda"],
            },
            "baselines": {"label_shuffle_null": {"components": ["all"]}},
        }
    )
    assert params.selection.components == ["basic", "Pairwise-MI", "soma_coupling"]


@pytest.mark.parametrize(
    "section, field",
    [
        ("selection", "components"),
        ("selection", "scopes"),
        ("selection", "signal_variants"),
    ],
)
def test_a_misspelled_token_raises_instead_of_disabling_the_selection(section, field):
    with pytest.raises(ValueError, match=rf"{section}\.{field} has unknown entries"):
        _information({section: {field: ["basci"]}})


def test_null_components_need_a_null_counterpart():
    with pytest.raises(ValueError, match=r"label_shuffle_null\.components"):
        _information({"baselines": {"label_shuffle_null": {"components": ["pid"]}}})
