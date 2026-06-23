"""Data-source loaders and activation capture for transformer replacement training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from dendritic_modeling.config import Config
from dendritic_modeling.networks.architectures.transformer import (
    DendriticFFNReplacement,
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
)
from dendritic_modeling.training.replacement_common import _load_tensor_payload
from dendritic_modeling.utils.hooks import ForwardHookRemovalMixin, register_hook_groups


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

    layer_indices = (
        train_cfg.layers or config.model.transformer_replacement.layers or [0]
    )
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


def _read_token_source(config: Config):
    train_cfg = config.training.transformer_replacement
    text_cfg = _to_plain_mapping(train_cfg.text)
    model_cfg = config.model.transformer_replacement
    tokenizer_name = str(text_cfg.get("tokenizer_name") or model_cfg.model_name)
    text_path = str(text_cfg.get("text_path", ""))

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for hf_text distillation") from exc

    if text_path:
        with open(text_path, encoding="utf-8") as handle:
            text = handle.read()
    else:
        dataset_name = str(text_cfg.get("dataset_name", ""))
        if not dataset_name:
            raise ValueError(
                "hf_text source requires either text.text_path or text.dataset_name"
            )
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required when hf_text uses text.dataset_name"
            ) from exc

        dataset_config = text_cfg.get("dataset_config") or None
        split = str(text_cfg.get("split", "train"))
        text_field = str(text_cfg.get("text_field", "text"))
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=bool(text_cfg.get("streaming", False)),
        )
        max_documents = text_cfg.get("max_documents")
        max_documents = None if max_documents is None else int(max_documents)
        docs = []
        for idx, row in enumerate(dataset):
            if max_documents is not None and idx >= max_documents:
                break
            value = str(row.get(text_field, ""))
            if value.strip():
                docs.append(value)
        if not docs:
            raise ValueError(
                f"Dataset {dataset_name!r} produced no non-empty {text_field!r} text"
            )
        text = "\n\n".join(docs)

    tokenizer_kwargs = _to_plain_mapping(text_cfg.get("tokenizer_kwargs", {}))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
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
    tokens: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
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

    layer_indices = (
        config.training.transformer_replacement.layers
        or config.model.transformer_replacement.layers
    )
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
        if _replacement_kind(config) in {
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
            replacement = DendriticFFNReplacement.from_mlp(mlp, **kwargs).to(
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
) -> nn.Module:
    """Load the configured causal LM onto one device/dtype for joint training."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("transformers is required for hf_text distillation") from exc

    model_cfg = config.model.transformer_replacement
    model_kwargs = _to_plain_mapping(model_cfg.model_kwargs)
    if "device_map" in model_kwargs:
        raise ValueError(
            "joint_lm_distillation expects a single-device model; remove device_map "
            "from model.transformer_replacement.model_kwargs"
        )
    model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name, **model_kwargs)
    model = model.to(device=device, dtype=dtype)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model
