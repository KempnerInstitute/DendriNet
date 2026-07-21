"""Data-source loaders and activation capture for transformer replacement training."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from dendritic_modeling.config import Config
from dendritic_modeling.networks.architectures.transformer import (
    DendriticFFNReplacement,
    GatedDendriticFFNReplacement,
    build_dendritic_ffn_kwargs_from_core_config,
    resolve_transformer_layers,
)
from dendritic_modeling.networks.architectures.transformer.utils import _get_attr_path
from dendritic_modeling.training._transformer_replacement.builders import (
    _make_replacement,
    _make_single_layer_ei_stack_replacement,
    _replacement_kind,
)
from dendritic_modeling.training._transformer_replacement.common import (
    DistillationUnit,
    _to_plain_mapping,
    resolve_configured_replacement_layers,
)
from dendritic_modeling.training.replacement_common import _load_tensor_payload
from dendritic_modeling.utils.hooks import ForwardHookRemovalMixin, register_hook_groups

from .token_stream import StreamingTokenBatchSource, resolve_data_files


def _sample_hidden_pairs(
    unit: DistillationUnit,
    *,
    n_samples: int,
    sequence_length: int,
    hidden_size: int,
    input_std: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> TensorDataset:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chunks_x: list[torch.Tensor] = []
    chunks_y: list[torch.Tensor] = []
    teacher = unit.teacher_mlp
    if teacher is None:
        raise ValueError("Synthetic hidden-pair sampling requires a teacher MLP")

    remaining = int(n_samples)
    with torch.no_grad():
        while remaining > 0:
            current = min(int(batch_size), remaining)
            x_cpu = torch.randn(
                current,
                int(sequence_length),
                int(hidden_size),
                generator=generator,
                dtype=torch.float32,
            )
            x_cpu = x_cpu * float(input_std)
            x = x_cpu.to(device=device, dtype=dtype)
            y = teacher(x).detach().to("cpu", dtype=torch.float32)
            chunks_x.append(x_cpu)
            chunks_y.append(y)
            remaining -= current

    return TensorDataset(torch.cat(chunks_x, dim=0), torch.cat(chunks_y, dim=0))


def _build_synthetic_datasets(
    units: Sequence[DistillationUnit],
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[TensorDataset], list[TensorDataset]]:
    train_cfg = config.training.transformer_replacement
    teacher_cfg = _to_plain_mapping(train_cfg.synthetic_teacher)
    hidden_size = int(train_cfg.hidden_size or 32)
    input_std = float(teacher_cfg.get("input_std", 1.0))
    base_seed = int(
        train_cfg.seed if train_cfg.seed is not None else config.experiment.seed
    )

    train_datasets = []
    valid_datasets = []
    for idx, unit in enumerate(units):
        train_datasets.append(
            _sample_hidden_pairs(
                unit,
                n_samples=train_cfg.train_samples,
                sequence_length=train_cfg.sequence_length,
                hidden_size=hidden_size,
                input_std=input_std,
                seed=base_seed + 17 * idx,
                device=device,
                dtype=dtype,
                batch_size=train_cfg.batch_size,
            )
        )
        valid_datasets.append(
            _sample_hidden_pairs(
                unit,
                n_samples=train_cfg.valid_samples,
                sequence_length=train_cfg.sequence_length,
                hidden_size=hidden_size,
                input_std=input_std,
                seed=base_seed + 10_003 + 17 * idx,
                device=device,
                dtype=dtype,
                batch_size=train_cfg.batch_size,
            )
        )
    return train_datasets, valid_datasets


def _load_tensor(path: str) -> Any:
    return _load_tensor_payload(
        path,
        empty_path_message="Hidden-cache tensor path is empty",
    )


def _select_layer_tensor(payload: Any, layer_index: int) -> torch.Tensor:
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, Mapping):
        for key in (layer_index, str(layer_index), f"layer_{layer_index}"):
            if key in payload:
                value = payload[key]
                if torch.is_tensor(value):
                    return value
        raise KeyError(f"No tensor for layer {layer_index} in hidden cache")
    raise TypeError("Hidden-cache payload must be a Tensor or dict of Tensors")


def _build_hidden_cache_units_and_datasets(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[DistillationUnit], list[TensorDataset], list[TensorDataset]]:
    train_cfg = config.training.transformer_replacement
    cache_cfg = _to_plain_mapping(train_cfg.hidden_cache)
    train_inputs = _load_tensor(str(cache_cfg.get("train_inputs", "")))
    train_targets = _load_tensor(str(cache_cfg.get("train_targets", "")))
    valid_inputs = _load_tensor(str(cache_cfg.get("valid_inputs", "")))
    valid_targets = _load_tensor(str(cache_cfg.get("valid_targets", "")))

    layer_indices = resolve_configured_replacement_layers(config, default=[0])
    units = []
    train_datasets = []
    valid_datasets = []
    for layer_index in layer_indices:
        x_train = _select_layer_tensor(train_inputs, int(layer_index)).float()
        y_train = _select_layer_tensor(train_targets, int(layer_index)).float()
        x_valid = _select_layer_tensor(valid_inputs, int(layer_index)).float()
        y_valid = _select_layer_tensor(valid_targets, int(layer_index)).float()
        hidden_size = int(x_train.shape[-1])
        replacement = _make_replacement(
            hidden_size=hidden_size,
            config=config,
            dtype=dtype,
            device=device,
        )
        units.append(
            DistillationUnit(layer_index=int(layer_index), replacement=replacement)
        )
        train_datasets.append(TensorDataset(x_train, y_train))
        valid_datasets.append(TensorDataset(x_valid, y_valid))
    return units, train_datasets, valid_datasets


def _extract_mlp_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(
        "Captured MLP output must be a Tensor or tuple/list starting with one"
    )


class _MLPIOCapture(ForwardHookRemovalMixin):
    def __init__(
        self,
        model: nn.Module,
        layer_indices: Sequence[int],
        *,
        layers_attr: str | None,
        mlp_attr: str,
    ):
        self.records: dict[int, dict[str, torch.Tensor]] = {}
        self.handles = []
        try:
            layers = resolve_transformer_layers(model, layers_attr=layers_attr)

            def _register_layer_hooks(layer_index_raw: int):
                layer_index = int(layer_index_raw)
                layer = layers[int(layer_index)]
                mlp = _get_attr_path(layer, mlp_attr)
                if not isinstance(mlp, nn.Module):
                    raise TypeError(
                        f"layer {layer_index} target {mlp_attr!r} is not an nn.Module"
                    )
                self.records[layer_index] = {}

                def _register_hook(hook_type: str):
                    if hook_type == "pre":
                        return [
                            mlp.register_forward_pre_hook(
                                self._make_pre_hook(layer_index)
                            )
                        ]
                    return [mlp.register_forward_hook(self._make_hook(layer_index))]

                return register_hook_groups(
                    ("pre", "forward"),
                    _register_hook,
                )

            self.handles = register_hook_groups(layer_indices, _register_layer_hooks)
        except Exception:
            self.close()
            raise

    def _make_pre_hook(self, layer_index: int):
        def hook(module: nn.Module, args: tuple[Any, ...]) -> None:
            del module
            self.records[layer_index]["input"] = args[0].detach()

        return hook

    def _make_hook(self, layer_index: int):
        def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            del module, args
            self.records[layer_index]["target"] = _extract_mlp_output(output).detach()

        return hook

    def clear(self) -> None:
        for value in self.records.values():
            value.clear()

    def close(self) -> None:
        self.remove_forward_hooks(self.handles)
        self.handles.clear()


def _flatten_tokenizer_input_ids(value: Any) -> list[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
        value = value[0]
    return [int(token_id) for token_id in value]


def _tokenize_documents(
    documents: Iterable[str],
    tokenizer: Any,
    *,
    max_tokens: int | None,
) -> torch.Tensor:
    token_ids: list[int] = []
    separator = getattr(tokenizer, "eos_token_id", None)
    for text in documents:
        encoded = tokenizer(str(text), add_special_tokens=False)
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise TypeError("Tokenizer output must contain an input_ids field")
        current = _flatten_tokenizer_input_ids(encoded["input_ids"])
        if current:
            token_ids.extend(current)
            if separator is not None:
                token_ids.append(int(separator))
        if max_tokens is not None and len(token_ids) >= int(max_tokens):
            del token_ids[int(max_tokens) :]
            break
    return torch.tensor(token_ids, dtype=torch.long)


def _load_text_dataset(
    text_cfg: Mapping[str, Any],
    *,
    split: str,
    data_files: Any,
    streaming: bool,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for hf_text dataset sources") from exc

    dataset_name = str(text_cfg.get("dataset_name", "") or "")
    normalized_files = resolve_data_files(data_files)
    if not dataset_name:
        if normalized_files is None:
            raise ValueError(
                "hf_text source requires text.text_path, text.dataset_name, "
                "or text.data_files"
            )
        dataset_name = "json"
    dataset_config = text_cfg.get("dataset_config") or None
    args = [dataset_name]
    if dataset_config is not None:
        args.append(str(dataset_config))
    kwargs: dict[str, Any] = {
        "split": str(split),
        "streaming": bool(streaming),
    }
    if normalized_files is not None:
        kwargs["data_files"] = normalized_files
    return load_dataset(*args, **kwargs)


def _iter_text_field(
    dataset: Iterable[Mapping[str, Any]],
    *,
    text_field: str,
    max_documents: int | None,
) -> Iterable[str]:
    for index, row in enumerate(dataset):
        if max_documents is not None and index >= int(max_documents):
            break
        value = str(row.get(text_field, ""))
        if value.strip():
            yield value


def _build_streaming_token_source(
    config: Config,
    tokenizer: Any,
    text_cfg: Mapping[str, Any],
) -> tuple[StreamingTokenBatchSource, torch.Tensor]:
    """Build a packed training stream and a bounded deterministic validation set."""
    train_cfg = config.training.transformer_replacement
    split = str(text_cfg.get("split", "train"))
    text_field = str(text_cfg.get("text_field", "text"))
    train_files = text_cfg.get("data_files")
    validation_files = text_cfg.get("validation_data_files")
    validation_split = str(text_cfg.get("validation_split", "") or "")
    validation_documents = max(1, int(text_cfg.get("validation_documents", 64)))
    separate_validation = validation_files not in (None, "", []) or bool(
        validation_split
    )

    validation_dataset = _load_text_dataset(
        text_cfg,
        split=validation_split or split,
        data_files=validation_files if separate_validation else train_files,
        streaming=True,
    )
    validation_max_tokens_raw = text_cfg.get("validation_max_tokens")
    if validation_max_tokens_raw is None:
        validation_max_tokens_raw = max(
            int(train_cfg.sequence_length) + 1,
            int(train_cfg.valid_samples) * int(train_cfg.sequence_length),
        )
    validation_tokens = _tokenize_documents(
        _iter_text_field(
            validation_dataset,
            text_field=text_field,
            max_documents=validation_documents,
        ),
        tokenizer,
        max_tokens=int(validation_max_tokens_raw),
    )
    min_tokens = max(4, int(train_cfg.sequence_length) + 1)
    if validation_tokens.numel() < min_tokens:
        raise ValueError(
            f"Need at least {min_tokens} validation tokens for hf_text distillation"
        )

    base_seed = int(
        text_cfg.get("shuffle_seed")
        if text_cfg.get("shuffle_seed") is not None
        else (train_cfg.seed or config.experiment.seed)
    )
    shuffle_buffer = int(text_cfg.get("shuffle_buffer_size", 10_000))
    max_documents_raw = text_cfg.get("max_documents")
    max_documents = None if max_documents_raw is None else int(max_documents_raw)
    world_size = int(text_cfg.get("world_size") or os.environ.get("WORLD_SIZE", 1))
    rank = int(text_cfg.get("rank") or os.environ.get("RANK", 0))
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid text-stream rank/world_size: {rank}/{world_size}")

    def document_factory(epoch: int) -> Iterable[str]:
        dataset = _load_text_dataset(
            text_cfg,
            split=split,
            data_files=train_files,
            streaming=True,
        )
        if not separate_validation and validation_documents:
            dataset = dataset.skip(validation_documents)
        if shuffle_buffer > 1:
            dataset = dataset.shuffle(
                seed=base_seed + int(epoch),
                buffer_size=shuffle_buffer,
            )
        if world_size > 1:
            dataset = dataset.shard(num_shards=world_size, index=rank)
        return _iter_text_field(
            dataset,
            text_field=text_field,
            max_documents=max_documents,
        )

    separator = getattr(tokenizer, "eos_token_id", None)
    stream = StreamingTokenBatchSource(
        document_factory,
        tokenizer,
        sequence_length=int(train_cfg.sequence_length),
        separator_token_id=separator,
        max_tokens_per_document=(
            None
            if text_cfg.get("max_tokens_per_document") is None
            else int(text_cfg["max_tokens_per_document"])
        ),
    )
    resume_stream_state = str(text_cfg.get("resume_stream_state", "") or "")
    resume_checkpoint = str(getattr(train_cfg, "resume_checkpoint", "") or "")
    if not resume_stream_state and resume_checkpoint:
        state_name = (
            "token_stream_checkpoint_state.json"
            if world_size == 1
            else f"token_stream_checkpoint_state.rank{rank}.json"
        )
        resume_stream_state = os.path.join(
            os.path.dirname(resume_checkpoint) or str(train_cfg.save_dir),
            state_name,
        )
    if resume_stream_state:
        stream.load_state(resume_stream_state)
    return stream, validation_tokens


def _read_token_source(
    config: Config,
) -> tuple[torch.Tensor | StreamingTokenBatchSource, torch.Tensor]:
    train_cfg = config.training.transformer_replacement
    text_cfg = _to_plain_mapping(train_cfg.text)
    model_cfg = config.model.transformer_replacement
    tokenizer_name = str(text_cfg.get("tokenizer_name") or model_cfg.model_name)
    text_path = str(text_cfg.get("text_path", ""))

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for hf_text distillation") from exc

    tokenizer_kwargs = _to_plain_mapping(text_cfg.get("tokenizer_kwargs", {}))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    if bool(text_cfg.get("streaming", False)) and not text_path:
        return _build_streaming_token_source(config, tokenizer, text_cfg)

    if text_path:
        with open(text_path, encoding="utf-8") as handle:
            text = handle.read()
    else:
        dataset_name = str(text_cfg.get("dataset_name", "") or "")
        if not dataset_name and not text_cfg.get("data_files"):
            raise ValueError(
                "hf_text source requires text.text_path, text.dataset_name, "
                "or text.data_files"
            )
        split = str(text_cfg.get("split", "train"))
        text_field = str(text_cfg.get("text_field", "text"))
        dataset = _load_text_dataset(
            text_cfg,
            split=split,
            data_files=text_cfg.get("data_files"),
            streaming=False,
        )
        max_documents = text_cfg.get("max_documents")
        max_documents = None if max_documents is None else int(max_documents)
        docs = list(
            _iter_text_field(
                dataset,
                text_field=text_field,
                max_documents=max_documents,
            )
        )
        if not docs:
            raise ValueError(
                f"Dataset {dataset_name or 'json'!r} produced no non-empty "
                f"{text_field!r} text"
            )
        text = "\n\n".join(docs)

    tokenized = tokenizer(text, return_tensors="pt")
    input_ids = tokenized["input_ids"].squeeze(0).long()
    max_tokens = text_cfg.get("max_tokens")
    if max_tokens is not None:
        input_ids = input_ids[: int(max_tokens)]
    min_tokens = max(4, int(train_cfg.sequence_length) + 1)
    if input_ids.numel() < min_tokens:
        raise ValueError(f"Need at least {min_tokens} tokens for hf_text distillation")
    split = max(int(0.9 * input_ids.numel()), min_tokens)
    return input_ids[:split], input_ids[max(0, split - train_cfg.sequence_length) :]


def _sample_token_batch(
    tokens: torch.Tensor | StreamingTokenBatchSource,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(tokens, StreamingTokenBatchSource):
        if int(sequence_length) != tokens.sequence_length:
            raise ValueError(
                "Requested sequence_length does not match the streaming source"
            )
        return tokens.next_batch(int(batch_size), device=device)
    max_start = tokens.numel() - int(sequence_length)
    if max_start <= 0:
        raise ValueError("Token source is shorter than sequence_length")
    starts = torch.randint(0, max_start, (int(batch_size),))
    batch = torch.stack([tokens[start : start + sequence_length] for start in starts])
    return batch.to(device=device)


def _load_hf_teacher_and_units(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, list[DistillationUnit]]:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("transformers is required for hf_text distillation") from exc

    model_cfg = config.model.transformer_replacement
    model_kwargs = _to_plain_mapping(model_cfg.model_kwargs)
    teacher = AutoModelForCausalLM.from_pretrained(model_cfg.model_name, **model_kwargs)
    if "device_map" not in model_kwargs:
        teacher = teacher.to(device=device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    layer_indices = resolve_configured_replacement_layers(config)
    if not layer_indices:
        raise ValueError("At least one transformer replacement layer is required")
    layers = resolve_transformer_layers(teacher, layers_attr=model_cfg.layers_attr)

    units: list[DistillationUnit] = []
    for layer_index in layer_indices:
        mlp = _get_attr_path(layers[int(layer_index)], model_cfg.target_module)
        if not isinstance(mlp, nn.Module):
            raise TypeError(
                f"layer {layer_index} target {model_cfg.target_module!r} "
                "is not an nn.Module"
            )
        kind = _replacement_kind(config)
        if kind in {"dense_mlp_control", "dense_control", "dense_mlp"}:
            raise ValueError("dense_mlp_control currently supports synthetic_mlp only")
        if kind in {
            "ei_stack",
            "layerwise_ei_stack",
            "einet_stack",
        }:
            param = next(mlp.parameters(), None)
            if param is None or param.ndim < 2:
                raise ValueError(
                    "Could not infer hidden size for EI-stack replacement from "
                    f"layer {layer_index}"
                )
            hidden_size = int(param.shape[1])
            replacement = _make_single_layer_ei_stack_replacement(
                hidden_size=hidden_size,
                config=config,
                dtype=dtype,
                device=device,
            )
        else:
            kwargs = build_dendritic_ffn_kwargs_from_core_config(
                config.model.core,
                transformer_replacement=config.model.transformer_replacement,
            )
            replacement_cls = (
                GatedDendriticFFNReplacement
                if kind in {"gated_dendritic_ffn", "gated_dendritic", "dendritic_glu"}
                else DendriticFFNReplacement
            )
            replacement = replacement_cls.from_mlp(mlp, **kwargs).to(
                device=device,
                dtype=dtype,
            )
        units.append(
            DistillationUnit(
                layer_index=int(layer_index),
                teacher_mlp=mlp,
                replacement=replacement,
            )
        )
    return teacher, units


def _load_hf_causal_lm_for_joint_training(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    initialization: str = "pretrained",
) -> nn.Module:
    """Load or initialize the causal LM for single-device joint training."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("transformers is required for hf_text distillation") from exc

    model_cfg = config.model.transformer_replacement
    model_kwargs = _to_plain_mapping(model_cfg.model_kwargs)
    if "device_map" in model_kwargs:
        raise ValueError(
            "joint_lm_distillation expects a single-device model; remove device_map "
            "from model.transformer_replacement.model_kwargs"
        )
    normalized = str(initialization).lower()
    if normalized in {"pretrained", "checkpoint"}:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_name,
            **model_kwargs,
        )
    elif normalized in {"from_config", "scratch", "random"}:
        config_kwargs = dict(model_kwargs)
        config_kwargs.pop("torch_dtype", None)
        hf_config = AutoConfig.from_pretrained(
            model_cfg.model_name,
            **config_kwargs,
        )
        model = AutoModelForCausalLM.from_config(
            hf_config,
            trust_remote_code=bool(config_kwargs.get("trust_remote_code", False)),
        )
    else:
        raise ValueError("student_initialization must be 'pretrained' or 'from_config'")
    model = model.to(device=device, dtype=dtype)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model
