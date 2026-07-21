"""Packed, restartable token streams for transformer replacement training."""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


class StreamingTokenBatchSource:
    """Pack a restartable document stream into fixed-length token batches.

    ``document_factory`` receives the zero-based stream epoch.  Recreating the
    factory with the same epoch must produce the same document order; this is
    what makes the compact state dict sufficient for an exact data-stream
    restart.
    """

    def __init__(
        self,
        document_factory: Callable[[int], Iterable[str]],
        tokenizer: Any,
        *,
        sequence_length: int,
        separator_token_id: int | None = None,
        max_tokens_per_document: int | None = None,
    ) -> None:
        if int(sequence_length) < 1:
            raise ValueError("sequence_length must be positive")
        if max_tokens_per_document is not None and int(max_tokens_per_document) < 1:
            raise ValueError("max_tokens_per_document must be positive when set")
        self.document_factory = document_factory
        self.tokenizer = tokenizer
        self.sequence_length = int(sequence_length)
        self.separator_token_id = (
            None if separator_token_id is None else int(separator_token_id)
        )
        self.max_tokens_per_document = (
            None if max_tokens_per_document is None else int(max_tokens_per_document)
        )
        self.epoch = 0
        self.documents_seen_in_epoch = 0
        self.sequences_emitted = 0
        self._token_buffer: list[int] = []
        self._document_iterator: Iterator[str] | None = None

    def _new_iterator(self) -> None:
        iterator = iter(self.document_factory(self.epoch))
        if self.documents_seen_in_epoch:
            iterator = itertools.islice(iterator, self.documents_seen_in_epoch, None)
        self._document_iterator = iterator

    def _tokenize_document(self, text: str) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=self.max_tokens_per_document is not None,
            max_length=self.max_tokens_per_document,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise TypeError("Tokenizer output must contain an input_ids field")
        token_ids = encoded["input_ids"]
        if torch.is_tensor(token_ids):
            token_ids = token_ids.detach().cpu().reshape(-1).tolist()
        elif (
            isinstance(token_ids, Sequence)
            and token_ids
            and isinstance(token_ids[0], Sequence)
        ):
            token_ids = token_ids[0]
        result = [int(token_id) for token_id in token_ids]
        if result and self.separator_token_id is not None:
            result.append(self.separator_token_id)
        return result

    def _append_next_document(self) -> bool:
        """Append one non-empty document, restarting the stream if necessary."""
        empty_epochs = 0
        while True:
            if self._document_iterator is None:
                self._new_iterator()
            assert self._document_iterator is not None
            try:
                text = next(self._document_iterator)
            except StopIteration:
                self.epoch += 1
                self.documents_seen_in_epoch = 0
                self._document_iterator = None
                empty_epochs += 1
                if empty_epochs >= 2:
                    return False
                continue
            self.documents_seen_in_epoch += 1
            token_ids = self._tokenize_document(str(text))
            if token_ids:
                self._token_buffer.extend(token_ids)
                return True

    def next_batch(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Return the next packed ``[batch, sequence]`` input-id tensor."""
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        required = int(batch_size) * self.sequence_length
        while len(self._token_buffer) < required:
            if not self._append_next_document():
                raise ValueError("Streaming text source produced no usable tokens")
        batch_tokens = self._token_buffer[:required]
        del self._token_buffer[:required]
        self.sequences_emitted += int(batch_size)
        return (
            torch.tensor(batch_tokens, dtype=torch.long)
            .reshape(int(batch_size), self.sequence_length)
            .to(device=device)
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the minimal exact-restart state for this packed stream."""
        return {
            "schema_version": 1,
            "epoch": self.epoch,
            "documents_seen_in_epoch": self.documents_seen_in_epoch,
            "sequences_emitted": self.sequences_emitted,
            "token_buffer": list(self._token_buffer),
            "sequence_length": self.sequence_length,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`."""
        state_sequence_length = int(state.get("sequence_length", self.sequence_length))
        if state_sequence_length != self.sequence_length:
            raise ValueError(
                "Token-stream state sequence_length does not match the current run"
            )
        self.epoch = int(state.get("epoch", 0))
        self.documents_seen_in_epoch = int(state.get("documents_seen_in_epoch", 0))
        self.sequences_emitted = int(state.get("sequences_emitted", 0))
        self._token_buffer = [int(value) for value in state.get("token_buffer", [])]
        self._document_iterator = None

    def save_state(self, path: str | Path) -> None:
        """Atomically save stream progress as portable JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state_dict(), indent=2) + "\n")
        os.replace(temporary, output)

    def load_state(self, path: str | Path) -> None:
        """Restore stream progress from :meth:`save_state`."""
        self.load_state_dict(json.loads(Path(path).read_text()))


def resolve_data_files(value: Any) -> str | list[str] | None:
    """Normalize a datasets ``data_files`` value without expanding huge globs."""
    if value is None or value == "":
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, Sequence):
        return [str(path) for path in value]
    raise TypeError("text.data_files must be a path, glob, or list of paths")


__all__ = ["StreamingTokenBatchSource", "resolve_data_files"]
