import sys
from types import SimpleNamespace

import torch

from dendritic_modeling.config import Config
from dendritic_modeling.training.transformer_replacement import (
    StreamingTokenBatchSource,
    _read_token_source,
    _sample_token_batch,
)


class _IntegerTokenizer:
    eos_token_id = 99

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [int(value) for value in str(text).split()]}


def test_streaming_token_batch_source_exact_restart(tmp_path):
    documents = ["1 2 3 4 5", "6 7 8", "9 10 11"]

    def factory(epoch):
        del epoch
        return iter(documents)

    source = StreamingTokenBatchSource(
        factory,
        _IntegerTokenizer(),
        sequence_length=4,
        separator_token_id=99,
    )
    first = source.next_batch(1, device="cpu")
    state_path = tmp_path / "token_stream_state.json"
    source.save_state(state_path)
    expected_next = source.next_batch(2, device="cpu")

    restored = StreamingTokenBatchSource(
        factory,
        _IntegerTokenizer(),
        sequence_length=4,
        separator_token_id=99,
    )
    restored.load_state(state_path)
    actual_next = restored.next_batch(2, device="cpu")

    assert first.tolist() == [[1, 2, 3, 4]]
    assert torch.equal(actual_next, expected_next)
    assert restored.state_dict() == source.state_dict()


def test_streaming_token_batch_source_restarts_finite_document_epochs():
    epochs = []

    def factory(epoch):
        epochs.append(epoch)
        return iter(["1 2 3"])

    source = StreamingTokenBatchSource(
        factory,
        _IntegerTokenizer(),
        sequence_length=4,
        separator_token_id=99,
    )

    batch = source.next_batch(2, device="cpu")

    assert batch.tolist() == [[1, 2, 3, 99], [1, 2, 3, 99]]
    assert epochs == [0, 1]
    assert source.epoch == 1


def test_read_token_source_streams_and_reserves_validation(monkeypatch):
    calls = []
    rows = [
        {"text": "1 2 3 4 5 6 7 8"},
        {"text": "10 11 12 13 14"},
        {"text": "20 21 22 23 24"},
    ]

    class FakeDataset:
        def __init__(self, values):
            self.values = list(values)

        def __iter__(self):
            return iter(self.values)

        def skip(self, count):
            calls.append(("skip", count))
            return FakeDataset(self.values[count:])

        def shuffle(self, *, seed, buffer_size):
            calls.append(("shuffle", seed, buffer_size))
            return FakeDataset(list(reversed(self.values)))

        def shard(self, *, num_shards, index):
            calls.append(("shard", num_shards, index))
            return FakeDataset(self.values[index::num_shards])

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("tokenizer", name, kwargs))
            return _IntegerTokenizer()

    def fake_load_dataset(*args, **kwargs):
        calls.append(("dataset", args, kwargs))
        return FakeDataset(rows)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    cfg = Config()
    cfg.experiment.seed = 17
    cfg.model.transformer_replacement.model_name = "local/tiny"
    train_cfg = cfg.training.transformer_replacement
    train_cfg.sequence_length = 4
    train_cfg.batch_size = 1
    train_cfg.valid_samples = 2
    train_cfg.text.update(
        {
            "dataset_name": "fake-corpus",
            "streaming": True,
            "validation_documents": 1,
            "shuffle_buffer_size": 32,
            "max_documents": 2,
        }
    )

    train_source, validation_tokens = _read_token_source(cfg)
    train_batch = _sample_token_batch(
        train_source,
        batch_size=1,
        sequence_length=4,
        device=torch.device("cpu"),
    )

    assert isinstance(train_source, StreamingTokenBatchSource)
    assert validation_tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert train_batch.tolist() == [[20, 21, 22, 23]]
    assert ("skip", 1) in calls
    assert ("shuffle", 17, 32) in calls
