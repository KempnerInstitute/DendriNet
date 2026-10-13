"""final_model.pt links an identical best checkpoint instead of copying it."""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from dendritic_modeling.networks.checkpoints import save_state_dict


def _state(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4)).state_dict()


def test_identical_weights_share_one_file(tmp_path):
    best = tmp_path / "standard_best_model.pt"
    torch.save(_state(0), best)
    final = tmp_path / "final_model.pt"

    assert save_state_dict(_state(0), final, identical_to=best) == "linked"
    assert os.path.samefile(best, final)
    loaded = torch.load(final, weights_only=True)
    assert all(torch.equal(loaded[k], v) for k, v in _state(0).items())


def test_different_weights_are_saved(tmp_path):
    best = tmp_path / "standard_best_model.pt"
    torch.save(_state(0), best)
    final = tmp_path / "final_model.pt"

    assert save_state_dict(_state(1), final, identical_to=best) == "saved"
    assert not os.path.samefile(best, final)
    loaded = torch.load(final, weights_only=True)
    assert all(torch.equal(loaded[k], v) for k, v in _state(1).items())


def test_a_refused_link_falls_back_to_saving(tmp_path, monkeypatch):
    best = tmp_path / "standard_best_model.pt"
    torch.save(_state(0), best)

    def refuse(*_args):
        raise OSError("no hard links here")

    monkeypatch.setattr(os, "link", refuse)
    final = tmp_path / "final_model.pt"
    assert save_state_dict(_state(0), final, identical_to=best) == "saved"
    assert not os.path.samefile(best, final)
    assert not list(tmp_path.glob(".final_model.pt.*"))


@pytest.mark.parametrize("identical_to", [None, "missing.pt"])
def test_without_a_best_checkpoint_the_state_is_saved(tmp_path, identical_to):
    target = None if identical_to is None else tmp_path / identical_to
    final = tmp_path / "final_model.pt"
    assert save_state_dict(_state(0), final, identical_to=target) == "saved"
    assert final.is_file()


def test_the_final_save_links_the_main_trainers_best_checkpoint(tmp_path):
    from dendritic_modeling.scripts.training.train_experiments import (
        _save_final_model_and_analysis,
    )

    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
    (tmp_path / "main_network").mkdir()
    best = tmp_path / "main_network" / "standard_best_model.pt"
    torch.save(model.state_dict(), best)

    _save_final_model_and_analysis(
        unwrapped_model=model,
        run_save_path=str(tmp_path),
        any_analysis_enabled=False,
        strategy="standard",
        train_ds=None,
        valid_ds=None,
        test_ds=None,
        analysis_manager=None,
        training_config=None,
        training_results={},
    )
    assert os.path.samefile(best, tmp_path / "final_model.pt")
