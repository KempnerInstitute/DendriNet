"""Distillation training for transformer dendritic FFN replacements."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import cycle
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.config import Config
from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.networks.architectures.transformer import (
    DendriticFFNReplacement,
    EIStackMLPSlot,
    LayerwiseEIStackReplacement,
    apply_transformer_replacement_config,
    build_dendritic_ffn_kwargs_from_core_config,
    build_ei_stack_kwargs_from_core_config,
    resolve_transformer_layers,
)
from dendritic_modeling.training.optimizers import create_optimizer

logger = logging.getLogger(__name__)


def _to_plain_mapping(obj: Any) -> dict[str, Any]:
    return to_plain_dict(obj)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> torch.dtype:
    aliases = {
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    normalized = str(dtype).lower()
    if normalized not in aliases:
        raise ValueError(f"Unknown dtype {dtype!r}; expected one of {sorted(aliases)}")
    return aliases[normalized]


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    normalized = str(name).lower()
    if normalized == "silu":
        return F.silu(x)
    if normalized == "gelu":
        return F.gelu(x)
    if normalized == "relu":
        return F.relu(x)
    if normalized in {"identity", "linear", "none"}:
        return x
    raise ValueError(f"Unsupported synthetic teacher activation {name!r}")


class SyntheticTransformerMLP(nn.Module):
    """Small DeepSeek/Llama-style teacher MLP for smoke distillation runs."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        kind: str = "gated_mlp",
        activation: str = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.kind = str(kind).lower()
        self.activation = activation
        if self.kind == "linear":
            self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        elif self.kind == "mlp":
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        elif self.kind == "gated_mlp":
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        else:
            raise ValueError("synthetic_teacher.kind must be linear, mlp, or gated_mlp")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "linear":
            return self.proj(x)
        if self.kind == "mlp":
            return self.down_proj(_activation(self.activation, self.up_proj(x)))
        return self.down_proj(
            _activation(self.activation, self.gate_proj(x)) * self.up_proj(x)
        )


@dataclass
class DistillationUnit:
    """One teacher/replacement pair."""

    layer_index: int
    replacement: nn.Module
    teacher_mlp: nn.Module | None = None


def _replacement_forward_modules(
    units: Sequence[DistillationUnit],
    train_cfg: Any,
    device: torch.device,
) -> list[nn.Module]:
    """Return modules used for forward passes, optionally DataParallel-wrapped."""
    replacements = [unit.replacement for unit in units]
    if not bool(getattr(train_cfg, "data_parallel", False)):
        return replacements
    if device.type != "cuda":
        logger.warning("data_parallel=True ignored because device is %s", device)
        return replacements

    available = torch.cuda.device_count()
    requested = [int(idx) for idx in getattr(train_cfg, "data_parallel_devices", [])]
    device_ids = requested if requested else list(range(available))
    device_ids = [idx for idx in device_ids if 0 <= idx < available]
    if len(device_ids) < 2:
        logger.warning(
            "data_parallel=True requested, but only %s CUDA device(s) are usable",
            available,
        )
        return replacements

    primary = torch.device(f"cuda:{device_ids[0]}")
    for module in replacements:
        module.to(primary)
    logger.info(
        "Using DataParallel for replacement modules on CUDA devices %s", device_ids
    )
    return [
        nn.DataParallel(module, device_ids=device_ids, output_device=device_ids[0])
        for module in replacements
    ]


@dataclass
class TransformerReplacementTrainingResult:
    """Summary returned by transformer replacement distillation."""

    train_losses: list[float]
    valid_losses: list[float]
    initial_valid_loss: float
    final_valid_loss: float
    best_valid_loss: float
    best_step: int
    save_dir: str
    layer_indices: list[int]

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TransformerReplacementBenchmarkResult:
    """Post-training comparison for a saved transformer replacement."""

    teacher_source: str
    layer_indices: list[int]
    checkpoint_dir: str
    initial: dict[str, float]
    trained: dict[str, float]
    teacher_dense_params: int | None
    dendritic_stored_params: int
    dendritic_active_params: int
    stored_reduction_factor: float | None
    active_reduction_factor: float | None
    initial_label: str = "initial"
    trained_label: str = "trained_replacement"

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    cosine_weight: float = 0.0,
) -> torch.Tensor:
    normalized = str(loss_name).lower()
    if normalized in {"mse", "l2"}:
        loss = F.mse_loss(prediction, target)
    elif normalized in {"smooth_l1", "huber"}:
        loss = F.smooth_l1_loss(prediction, target)
    else:
        raise ValueError("transformer replacement loss must be mse or smooth_l1")

    if cosine_weight > 0:
        pred_flat = prediction.reshape(-1, prediction.shape[-1])
        tgt_flat = target.reshape(-1, target.shape[-1])
        cosine_loss = 1.0 - F.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean()
        loss = loss + float(cosine_weight) * cosine_loss
    return loss


def _replacement_kind(config: Config) -> str:
    replacement_kwargs = _to_plain_mapping(
        config.model.transformer_replacement.replacement_kwargs
    )
    return str(
        replacement_kwargs.get("kind", replacement_kwargs.get("type", "dendritic_ffn"))
    ).lower()


def _make_single_layer_ei_stack_replacement(
    *,
    hidden_size: int,
    config: Config,
    dtype: torch.dtype,
    device: torch.device,
) -> EIStackMLPSlot:
    kwargs = build_ei_stack_kwargs_from_core_config(
        config.model.core,
        transformer_replacement=config.model.transformer_replacement,
        num_layers=1,
    )
    stack = LayerwiseEIStackReplacement(hidden_size=hidden_size, **kwargs)
    stack = stack.to(device=device, dtype=dtype)
    return EIStackMLPSlot(stack, 0)


def _make_replacement(
    *,
    hidden_size: int,
    config: Config,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    if _replacement_kind(config) in {"ei_stack", "layerwise_ei_stack", "einet_stack"}:
        return _make_single_layer_ei_stack_replacement(
            hidden_size=hidden_size,
            config=config,
            dtype=dtype,
            device=device,
        )

    kwargs = build_dendritic_ffn_kwargs_from_core_config(
        config.model.core,
        transformer_replacement=config.model.transformer_replacement,
    )
    replacement = DendriticFFNReplacement(hidden_size=hidden_size, **kwargs)
    return replacement.to(device=device, dtype=dtype)


def _make_synthetic_units(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[DistillationUnit]:
    train_cfg = config.training.transformer_replacement
    teacher_cfg = _to_plain_mapping(train_cfg.synthetic_teacher)
    hidden_size = int(train_cfg.hidden_size or 32)
    intermediate_size = int(train_cfg.intermediate_size or hidden_size * 4)
    layer_indices = (
        train_cfg.layers or config.model.transformer_replacement.layers or [0]
    )

    units: list[DistillationUnit] = []
    for layer_index in layer_indices:
        teacher = SyntheticTransformerMLP(
            hidden_size,
            intermediate_size,
            kind=str(teacher_cfg.get("kind", "gated_mlp")),
            activation=str(teacher_cfg.get("activation", "silu")),
            bias=bool(teacher_cfg.get("bias", False)),
        ).to(device=device, dtype=dtype)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

        replacement = _make_replacement(
            hidden_size=hidden_size,
            config=config,
            dtype=dtype,
            device=device,
        )
        units.append(
            DistillationUnit(
                layer_index=int(layer_index),
                teacher_mlp=teacher,
                replacement=replacement,
            )
        )
    return units


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
    if not path:
        raise ValueError("Hidden-cache tensor path is empty")
    return torch.load(path, map_location="cpu")


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


class _MLPIOCapture:
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
        layers = resolve_transformer_layers(model, layers_attr=layers_attr)
        for layer_index in layer_indices:
            layer = layers[int(layer_index)]
            mlp = getattr(layer, mlp_attr)
            self.records[int(layer_index)] = {}
            self.handles.append(
                mlp.register_forward_pre_hook(self._make_pre_hook(int(layer_index)))
            )
            self.handles.append(
                mlp.register_forward_hook(self._make_hook(int(layer_index)))
            )

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
        for handle in self.handles:
            handle.remove()
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
        mlp = getattr(layers[int(layer_index)], model_cfg.target_module)
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


def _records_to_units(records: Sequence[Any]) -> list[DistillationUnit]:
    return [
        DistillationUnit(
            layer_index=int(record.layer_index),
            replacement=record.replacement,
            teacher_mlp=None,
        )
        for record in records
    ]


def _remap_single_layer_stack_state_dict(
    replacement: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    """Map a single-layer EI-stack checkpoint into one slot of a shared stack."""
    if not isinstance(replacement, EIStackMLPSlot):
        return None

    slot_index = int(replacement.layer_index)
    remapped: dict[str, torch.Tensor] = {}
    prefix_map = {
        "stack.layers.0.": f"stack.layers.{slot_index}.",
        "stack.output_projections.0.": f"stack.output_projections.{slot_index}.",
        "stack.pre_norm.0.": f"stack.pre_norm.{slot_index}.",
    }
    for key, value in state_dict.items():
        for old_prefix, new_prefix in prefix_map.items():
            if key.startswith(old_prefix):
                remapped[f"{new_prefix}{key[len(old_prefix):]}"] = value
                break

    return remapped or None


def _load_replacement_state_dict(
    replacement: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Load a replacement checkpoint, including layerwise EI-stack slots."""
    remapped = _remap_single_layer_stack_state_dict(replacement, state_dict)
    if remapped is not None and set(state_dict) != set(replacement.state_dict()):
        _copy_replacement_state_dict(replacement, remapped)
        return

    try:
        replacement.load_state_dict(state_dict)
        return
    except RuntimeError as exc:
        if remapped is None:
            raise exc

    _copy_replacement_state_dict(replacement, remapped)


def _copy_replacement_state_dict(
    replacement: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    current_state = replacement.state_dict()
    missing = [key for key in state_dict if key not in current_state]
    if missing:
        raise RuntimeError(
            "EI-stack checkpoint keys do not match the replacement slot: "
            f"{missing[:5]}"
        )
    with torch.no_grad():
        for key, value in state_dict.items():
            target = current_state[key]
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def _load_replacement_record_checkpoints(
    records: Sequence[Any],
    checkpoint_dir: str,
    *,
    device: torch.device,
) -> None:
    if not checkpoint_dir:
        return
    for record in records:
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"layer_{int(record.layer_index)}_replacement.pt",
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=device)
        state_dict = payload.get("state_dict", payload)
        _load_replacement_state_dict(record.replacement, state_dict)


def _freeze_except_replacements(
    model: nn.Module,
    records: Sequence[Any],
) -> list[nn.Parameter]:
    for param in model.parameters():
        param.requires_grad_(False)
    trainable: list[nn.Parameter] = []
    seen: set[int] = set()
    for record in records:
        for param in record.replacement.parameters():
            param.requires_grad_(True)
            param_id = id(param)
            if param_id not in seen:
                seen.add(param_id)
                trainable.append(param)
    if not trainable:
        raise ValueError("No replacement parameters are trainable")
    return trainable


def _resolve_joint_hidden_layers(
    config: Config,
) -> list[int]:
    train_cfg = config.training.transformer_replacement
    configured = list(getattr(train_cfg, "hidden_loss_layers", []) or [])
    if configured:
        return [int(layer) for layer in configured]
    return [
        int(layer)
        for layer in (
            train_cfg.layers or config.model.transformer_replacement.layers or []
        )
    ]


def _joint_lm_distillation_loss(
    student_output: Any,
    teacher_output: Any | None,
    *,
    hidden_layers: Sequence[int],
    lm_loss_weight: float,
    kl_loss_weight: float,
    kl_temperature: float,
    hidden_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if student_output.loss is None:
        raise ValueError("Student output did not include a language-modeling loss")

    total = student_output.loss * float(lm_loss_weight)
    components = {
        "lm": float(student_output.loss.detach().float().item()),
        "kl": 0.0,
        "hidden": 0.0,
    }

    if kl_loss_weight > 0:
        if teacher_output is None:
            raise ValueError("KL loss requires teacher_output")
        temperature = max(float(kl_temperature), 1e-6)
        student_logits = student_output.logits[:, :-1, :].float() / temperature
        teacher_logits = teacher_output.logits[:, :-1, :].float() / temperature
        vocab = student_logits.shape[-1]
        student_flat = student_logits.reshape(-1, vocab)
        teacher_flat = teacher_logits.reshape(-1, vocab)
        kl = F.kl_div(
            F.log_softmax(student_flat, dim=-1),
            F.softmax(teacher_flat, dim=-1),
            reduction="batchmean",
        ) * (temperature * temperature)
        total = total + float(kl_loss_weight) * kl
        components["kl"] = float(kl.detach().item())

    if hidden_loss_weight > 0:
        if teacher_output is None:
            raise ValueError("Hidden loss requires teacher_output")
        if student_output.hidden_states is None or teacher_output.hidden_states is None:
            raise ValueError("Hidden loss requires output_hidden_states=True")
        hidden_terms = []
        for layer in hidden_layers:
            hidden_index = int(layer) + 1
            hidden_terms.append(
                F.mse_loss(
                    student_output.hidden_states[hidden_index].float(),
                    teacher_output.hidden_states[hidden_index].float(),
                )
            )
        if hidden_terms:
            hidden_loss = torch.stack(hidden_terms).mean()
            total = total + float(hidden_loss_weight) * hidden_loss
            components["hidden"] = float(hidden_loss.detach().item())

    components["total"] = float(total.detach().float().item())
    return total, components


def _evaluate_tensor_datasets(
    units: Sequence[DistillationUnit],
    datasets: Sequence[TensorDataset],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    loss_name: str,
    cosine_weight: float,
    forward_modules: Sequence[nn.Module] | None = None,
) -> float:
    total = 0.0
    count = 0
    modules = (
        list(forward_modules)
        if forward_modules is not None
        else [unit.replacement for unit in units]
    )
    loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for dataset in datasets
    ]
    with torch.no_grad():
        for module, loader in zip(modules, loaders):
            module.eval()
            for x, target in loader:
                x = x.to(device=device, dtype=dtype)
                target = target.to(device=device, dtype=dtype)
                loss = _compute_distillation_loss(
                    module(x),
                    target,
                    loss_name=loss_name,
                    cosine_weight=cosine_weight,
                )
                total += float(loss.item())
                count += 1
    return total / max(count, 1)


def _compute_prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    pred_flat = prediction.reshape(-1, prediction.shape[-1]).float()
    target_flat = target.reshape(-1, target.shape[-1]).float()
    error = pred_flat - target_flat
    mse = float(error.square().mean().item())
    mae = float(error.abs().mean().item())
    target_power = float(target_flat.square().mean().item())
    target_mean = target_flat.mean(dim=0, keepdim=True)
    total_var = float((target_flat - target_mean).square().sum().item())
    sse = float(error.square().sum().item())
    cosine = float(F.cosine_similarity(pred_flat, target_flat, dim=-1).mean().item())
    return {
        "mse": mse,
        "mae": mae,
        "relative_mse": mse / max(target_power, 1e-12),
        "cosine": cosine,
        "r2": 1.0 - sse / max(total_var, 1e-12),
    }


def _evaluate_tensor_dataset_metrics(
    units: Sequence[DistillationUnit],
    datasets: Sequence[TensorDataset],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    predictions = []
    targets = []
    loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for dataset in datasets
    ]
    with torch.no_grad():
        for unit, loader in zip(units, loaders):
            unit.replacement.eval()
            for x, target in loader:
                x = x.to(device=device, dtype=dtype)
                predictions.append(unit.replacement(x).detach().cpu())
                targets.append(target.detach().cpu())
    if not predictions:
        raise ValueError("No benchmark batches were evaluated")
    return _compute_prediction_metrics(
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
    )


def _synthetic_teacher_parameter_count(config: Config) -> int | None:
    train_cfg = config.training.transformer_replacement
    if str(train_cfg.teacher_source).lower() != "synthetic_mlp":
        return None
    teacher_cfg = _to_plain_mapping(train_cfg.synthetic_teacher)
    hidden_size = int(train_cfg.hidden_size or 32)
    intermediate_size = int(train_cfg.intermediate_size or hidden_size * 4)
    kind = str(teacher_cfg.get("kind", "gated_mlp")).lower()
    bias = bool(teacher_cfg.get("bias", False))
    if kind == "linear":
        count = hidden_size * hidden_size
        return count + (hidden_size if bias else 0)
    if kind == "mlp":
        count = hidden_size * intermediate_size + intermediate_size * hidden_size
        return count + (intermediate_size + hidden_size if bias else 0)
    if kind == "gated_mlp":
        count = 3 * hidden_size * intermediate_size
        return count + (2 * intermediate_size + hidden_size if bias else 0)
    return None


def _evaluate_lm_loss_on_tokens(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    max_batches: int,
) -> dict[str, float]:
    """Evaluate causal-LM loss on deterministic windows from ``tokens``."""
    model.eval()
    input_device = next(model.parameters()).device
    max_start = int(tokens.numel()) - int(sequence_length)
    if max_start <= 0:
        raise ValueError("Token source is shorter than sequence_length")

    losses = []
    n_tokens = 0
    with torch.no_grad():
        for batch_idx in range(max(1, int(max_batches))):
            starts = [
                (batch_idx * int(batch_size) + offset) * int(sequence_length)
                for offset in range(int(batch_size))
            ]
            starts = [start % max_start for start in starts]
            input_ids = torch.stack(
                [tokens[start : start + int(sequence_length)] for start in starts]
            ).to(device=input_device)
            output = model(input_ids=input_ids, labels=input_ids)
            loss = output.loss.detach().float()
            losses.append(loss.cpu())
            n_tokens += int(input_ids.numel())
    mean_loss = float(torch.stack(losses).mean().item())
    perplexity = float(torch.exp(torch.tensor(min(mean_loss, 50.0))).item())
    return {
        "lm_loss": mean_loss,
        "perplexity": perplexity,
        "n_tokens": float(n_tokens),
    }


def _benchmark_hf_text_replacement(
    config: Config,
    *,
    checkpoint_dir: str,
    device: torch.device,
) -> TransformerReplacementBenchmarkResult:
    """Evaluate original vs patched causal-LM loss on the configured text."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("transformers is required for hf_text benchmarking") from exc

    train_cfg = config.training.transformer_replacement
    model_cfg = config.model.transformer_replacement
    model_kwargs = _to_plain_mapping(model_cfg.model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name, **model_kwargs)
    if "device_map" not in model_kwargs:
        model = model.to(device=device)
    _, valid_tokens = _read_token_source(config)
    eval_batches = max(
        1, int(train_cfg.valid_samples) // max(1, int(train_cfg.batch_size))
    )

    original_metrics = _evaluate_lm_loss_on_tokens(
        model,
        valid_tokens,
        batch_size=train_cfg.batch_size,
        sequence_length=train_cfg.sequence_length,
        max_batches=eval_batches,
    )

    records = apply_transformer_replacement_config(
        model,
        config.model.transformer_replacement,
        config.model.core,
    )
    if not records:
        raise ValueError("No transformer replacement records were created")

    for record in records:
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"layer_{record.layer_index}_replacement.pt",
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(
            checkpoint_path, map_location=next(model.parameters()).device
        )
        state_dict = payload.get("state_dict", payload)
        _load_replacement_state_dict(record.replacement, state_dict)

    patched_metrics = _evaluate_lm_loss_on_tokens(
        model,
        valid_tokens,
        batch_size=train_cfg.batch_size,
        sequence_length=train_cfg.sequence_length,
        max_batches=eval_batches,
    )

    teacher_dense = sum(
        sum(param.numel() for param in record.original_mlp.parameters())
        for record in records
    )
    estimates = [record.replacement.parameter_estimate() for record in records]
    stored = sum(int(estimate["stored_total"]) for estimate in estimates)
    active = sum(int(estimate["active_total"]) for estimate in estimates)

    result = TransformerReplacementBenchmarkResult(
        teacher_source="hf_text",
        layer_indices=[int(record.layer_index) for record in records],
        checkpoint_dir=checkpoint_dir,
        initial=original_metrics,
        trained=patched_metrics,
        teacher_dense_params=teacher_dense,
        dendritic_stored_params=stored,
        dendritic_active_params=active,
        stored_reduction_factor=teacher_dense / max(stored, 1),
        active_reduction_factor=teacher_dense / max(active, 1),
        initial_label="original_teacher_model",
        trained_label="trained_replacement_model",
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(
        os.path.join(checkpoint_dir, "benchmark_metrics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result.asdict(), handle, indent=2)
    return result


def _save_result(
    result: TransformerReplacementTrainingResult,
    units: Sequence[DistillationUnit],
    *,
    save_replacements: bool,
) -> None:
    os.makedirs(result.save_dir, exist_ok=True)
    with open(
        os.path.join(result.save_dir, "metrics.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result.asdict(), handle, indent=2)
    if not save_replacements:
        return
    for unit in units:
        torch.save(
            {
                "layer_index": unit.layer_index,
                "state_dict": unit.replacement.state_dict(),
                "parameter_estimate": unit.replacement.parameter_estimate(),
            },
            os.path.join(result.save_dir, f"layer_{unit.layer_index}_replacement.pt"),
        )


def _train_tensor_dataset_source(
    units: Sequence[DistillationUnit],
    train_datasets: Sequence[TensorDataset],
    valid_datasets: Sequence[TensorDataset],
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TransformerReplacementTrainingResult:
    train_cfg = config.training.transformer_replacement
    loaders = [
        cycle(
            DataLoader(
                dataset,
                batch_size=train_cfg.batch_size,
                shuffle=True,
                drop_last=False,
            )
        )
        for dataset in train_datasets
    ]
    forward_modules = _replacement_forward_modules(units, train_cfg, device)
    trainable_modules = nn.ModuleList(forward_modules)
    optimizer = create_optimizer(
        trainable_modules.parameters(), config.training.main.optimizer
    )
    scaler = GradScaler(
        "cuda", enabled=bool(train_cfg.use_amp and device.type == "cuda")
    )

    initial_valid = _evaluate_tensor_datasets(
        units,
        valid_datasets,
        batch_size=train_cfg.batch_size,
        device=device,
        dtype=dtype,
        loss_name=train_cfg.loss,
        cosine_weight=train_cfg.cosine_weight,
        forward_modules=forward_modules,
    )
    best_valid = initial_valid
    best_step = 0
    train_losses: list[float] = []
    valid_losses: list[float] = [initial_valid]

    for step in range(1, int(train_cfg.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device, dtype=dtype)
        for module, loader in zip(forward_modules, loaders):
            module.train()
            x, target = next(loader)
            x = x.to(device=device, dtype=dtype)
            target = target.to(device=device, dtype=dtype)
            with autocast("cuda", enabled=scaler.is_enabled()):
                prediction = module(x)
                loss = _compute_distillation_loss(
                    prediction,
                    target,
                    loss_name=train_cfg.loss,
                    cosine_weight=train_cfg.cosine_weight,
                )
            step_loss = step_loss + loss / max(len(units), 1)
        scaler.scale(step_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(float(step_loss.detach().item()))

        if step % int(train_cfg.eval_every) == 0 or step == int(train_cfg.max_steps):
            valid = _evaluate_tensor_datasets(
                units,
                valid_datasets,
                batch_size=train_cfg.batch_size,
                device=device,
                dtype=dtype,
                loss_name=train_cfg.loss,
                cosine_weight=train_cfg.cosine_weight,
                forward_modules=forward_modules,
            )
            valid_losses.append(valid)
            if valid < best_valid:
                best_valid = valid
                best_step = step
        if step % int(train_cfg.log_every) == 0 or step == 1:
            logger.info(
                "transformer replacement distillation step=%s/%s train_loss=%.6f valid_loss=%.6f",
                step,
                train_cfg.max_steps,
                train_losses[-1],
                valid_losses[-1],
            )

    result = TransformerReplacementTrainingResult(
        train_losses=train_losses,
        valid_losses=valid_losses,
        initial_valid_loss=initial_valid,
        final_valid_loss=valid_losses[-1],
        best_valid_loss=best_valid,
        best_step=best_step,
        save_dir=train_cfg.save_dir,
        layer_indices=[unit.layer_index for unit in units],
    )
    _save_result(result, units, save_replacements=train_cfg.save_replacements)
    return result


def benchmark_saved_transformer_replacement(
    config: Config,
    *,
    checkpoint_dir: str | None = None,
) -> TransformerReplacementBenchmarkResult:
    """Compare untrained and trained dendritic replacements against the teacher.

    The benchmark reconstructs the configured teacher data source, evaluates the
    initial dendritic replacement, loads ``layer_<idx>_replacement.pt`` files,
    and evaluates the trained replacement on the same validation hidden states.
    """
    train_cfg = config.training.transformer_replacement
    if not train_cfg.enabled:
        raise ValueError("training.transformer_replacement.enabled must be true")

    device = _resolve_device(train_cfg.device)
    dtype = _resolve_dtype(train_cfg.dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    seed = int(train_cfg.seed if train_cfg.seed is not None else config.experiment.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ckpt_dir = checkpoint_dir or train_cfg.save_dir
    source = str(train_cfg.teacher_source).lower()
    if source == "hf_text":
        return _benchmark_hf_text_replacement(
            config,
            checkpoint_dir=ckpt_dir,
            device=device,
        )

    if source == "synthetic_mlp":
        units = _make_synthetic_units(config, device=device, dtype=dtype)
        _, valid_datasets = _build_synthetic_datasets(
            units,
            config,
            device=device,
            dtype=dtype,
        )
    elif source == "hidden_cache":
        units, _, valid_datasets = _build_hidden_cache_units_and_datasets(
            config,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(
            "Post-training benchmark currently supports synthetic_mlp and "
            "hidden_cache teacher sources. Use hidden_cache for large HF models."
        )

    initial_metrics = _evaluate_tensor_dataset_metrics(
        units,
        valid_datasets,
        batch_size=train_cfg.batch_size,
        device=device,
        dtype=dtype,
    )

    for unit in units:
        checkpoint_path = os.path.join(
            ckpt_dir,
            f"layer_{unit.layer_index}_replacement.pt",
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=device)
        state_dict = payload.get("state_dict", payload)
        _load_replacement_state_dict(unit.replacement, state_dict)

    trained_metrics = _evaluate_tensor_dataset_metrics(
        units,
        valid_datasets,
        batch_size=train_cfg.batch_size,
        device=device,
        dtype=dtype,
    )

    estimates = [unit.replacement.parameter_estimate() for unit in units]
    stored = sum(int(estimate["stored_total"]) for estimate in estimates)
    active = sum(int(estimate["active_total"]) for estimate in estimates)
    teacher_dense = _synthetic_teacher_parameter_count(config)
    if teacher_dense is not None:
        teacher_dense *= max(len(units), 1)

    result = TransformerReplacementBenchmarkResult(
        teacher_source=source,
        layer_indices=[unit.layer_index for unit in units],
        checkpoint_dir=ckpt_dir,
        initial=initial_metrics,
        trained=trained_metrics,
        teacher_dense_params=teacher_dense,
        dendritic_stored_params=stored,
        dendritic_active_params=active,
        stored_reduction_factor=(
            None if teacher_dense is None else teacher_dense / max(stored, 1)
        ),
        active_reduction_factor=(
            None if teacher_dense is None else teacher_dense / max(active, 1)
        ),
    )

    os.makedirs(ckpt_dir, exist_ok=True)
    with open(
        os.path.join(ckpt_dir, "benchmark_metrics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result.asdict(), handle, indent=2)
    return result


def _evaluate_hf_text(
    teacher: nn.Module,
    units: Sequence[DistillationUnit],
    tokens: torch.Tensor,
    capture: _MLPIOCapture,
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_batches: int = 4,
    forward_modules: Sequence[nn.Module] | None = None,
) -> float:
    train_cfg = config.training.transformer_replacement
    input_device = next(teacher.parameters()).device
    modules = (
        list(forward_modules)
        if forward_modules is not None
        else [unit.replacement for unit in units]
    )
    total = 0.0
    count = 0
    with torch.no_grad():
        for _ in range(max(1, n_batches)):
            capture.clear()
            input_ids = _sample_token_batch(
                tokens,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                device=input_device,
            )
            teacher(input_ids=input_ids)
            for unit, module in zip(units, modules):
                values = capture.records[unit.layer_index]
                x = values["input"].to(device=device, dtype=dtype)
                target = values["target"].to(device=device, dtype=dtype)
                module.eval()
                loss = _compute_distillation_loss(
                    module(x),
                    target,
                    loss_name=train_cfg.loss,
                    cosine_weight=train_cfg.cosine_weight,
                )
                total += float(loss.item())
                count += 1
    return total / max(count, 1)


def _train_hf_text_source(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TransformerReplacementTrainingResult:
    train_cfg = config.training.transformer_replacement
    model_cfg = config.model.transformer_replacement
    teacher, units = _load_hf_teacher_and_units(config, device=device, dtype=dtype)
    train_tokens, valid_tokens = _read_token_source(config)
    capture = _MLPIOCapture(
        teacher,
        [unit.layer_index for unit in units],
        layers_attr=model_cfg.layers_attr,
        mlp_attr=model_cfg.target_module,
    )
    forward_modules = _replacement_forward_modules(units, train_cfg, device)
    trainable_modules = nn.ModuleList(forward_modules)
    optimizer = create_optimizer(
        trainable_modules.parameters(), config.training.main.optimizer
    )
    scaler = GradScaler(
        "cuda", enabled=bool(train_cfg.use_amp and device.type == "cuda")
    )
    input_device = next(teacher.parameters()).device

    initial_valid = _evaluate_hf_text(
        teacher,
        units,
        valid_tokens,
        capture,
        config,
        device=device,
        dtype=dtype,
        forward_modules=forward_modules,
    )
    best_valid = initial_valid
    best_step = 0
    train_losses: list[float] = []
    valid_losses: list[float] = [initial_valid]

    try:
        for step in range(1, int(train_cfg.max_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            capture.clear()
            input_ids = _sample_token_batch(
                train_tokens,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                device=input_device,
            )
            with torch.no_grad():
                teacher(input_ids=input_ids)

            step_loss = torch.zeros((), device=device, dtype=dtype)
            for unit, module in zip(units, forward_modules):
                values = capture.records[unit.layer_index]
                x = values["input"].to(device=device, dtype=dtype)
                target = values["target"].to(device=device, dtype=dtype)
                module.train()
                with autocast("cuda", enabled=scaler.is_enabled()):
                    loss = _compute_distillation_loss(
                        module(x),
                        target,
                        loss_name=train_cfg.loss,
                        cosine_weight=train_cfg.cosine_weight,
                    )
                step_loss = step_loss + loss / max(len(units), 1)

            scaler.scale(step_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(step_loss.detach().item()))

            if step % int(train_cfg.eval_every) == 0 or step == int(
                train_cfg.max_steps
            ):
                valid = _evaluate_hf_text(
                    teacher,
                    units,
                    valid_tokens,
                    capture,
                    config,
                    device=device,
                    dtype=dtype,
                    forward_modules=forward_modules,
                )
                valid_losses.append(valid)
                if valid < best_valid:
                    best_valid = valid
                    best_step = step
            if step % int(train_cfg.log_every) == 0 or step == 1:
                logger.info(
                    "hf_text dendritic distillation step=%s/%s train_loss=%.6f valid_loss=%.6f",
                    step,
                    train_cfg.max_steps,
                    train_losses[-1],
                    valid_losses[-1],
                )
    finally:
        capture.close()

    result = TransformerReplacementTrainingResult(
        train_losses=train_losses,
        valid_losses=valid_losses,
        initial_valid_loss=initial_valid,
        final_valid_loss=valid_losses[-1],
        best_valid_loss=best_valid,
        best_step=best_step,
        save_dir=train_cfg.save_dir,
        layer_indices=[unit.layer_index for unit in units],
    )
    _save_result(result, units, save_replacements=train_cfg.save_replacements)
    return result


def _train_joint_hf_text_source(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TransformerReplacementTrainingResult:
    """Train all patched dendritic MLPs through the full LM objective."""
    train_cfg = config.training.transformer_replacement
    model_cfg = config.model.transformer_replacement

    train_tokens, valid_tokens = _read_token_source(config)
    teacher_needed = bool(
        float(train_cfg.kl_loss_weight) > 0 or float(train_cfg.hidden_loss_weight) > 0
    )
    teacher = (
        _load_hf_causal_lm_for_joint_training(config, device=device, dtype=dtype)
        if teacher_needed
        else None
    )
    if teacher is not None:
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

    student = _load_hf_causal_lm_for_joint_training(
        config,
        device=device,
        dtype=dtype,
    )
    records = apply_transformer_replacement_config(
        student,
        model_cfg,
        config.model.core,
    )
    if not records:
        raise ValueError("No transformer replacement records were created")
    warm_start_dir = str(getattr(train_cfg, "warm_start_dir", "") or "")
    if warm_start_dir:
        logger.info(
            "Loading dendritic replacement warm-start from %s",
            warm_start_dir,
        )
        _load_replacement_record_checkpoints(
            records,
            warm_start_dir,
            device=device,
        )

    trainable_params = _freeze_except_replacements(student, records)
    if bool(getattr(train_cfg, "gradient_checkpointing", False)):
        if hasattr(student, "gradient_checkpointing_enable"):
            student.gradient_checkpointing_enable()
        student.train()
    else:
        student.eval()
    for record in records:
        record.replacement.train()

    optimizer = create_optimizer(trainable_params, config.training.main.optimizer)
    scaler = GradScaler(
        "cuda", enabled=bool(train_cfg.use_amp and device.type == "cuda")
    )

    eval_batches = max(
        1, int(train_cfg.valid_samples) // max(1, int(train_cfg.batch_size))
    )
    initial_metrics = _evaluate_lm_loss_on_tokens(
        student,
        valid_tokens,
        batch_size=train_cfg.batch_size,
        sequence_length=train_cfg.sequence_length,
        max_batches=eval_batches,
    )
    initial_valid = float(initial_metrics["lm_loss"])
    best_valid = initial_valid
    best_step = 0
    train_losses: list[float] = []
    valid_losses: list[float] = [initial_valid]
    hidden_layers = _resolve_joint_hidden_layers(config)
    input_device = next(student.parameters()).device

    for step in range(1, int(train_cfg.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        input_ids = _sample_token_batch(
            train_tokens,
            batch_size=train_cfg.batch_size,
            sequence_length=train_cfg.sequence_length,
            device=input_device,
        )
        output_hidden = float(train_cfg.hidden_loss_weight) > 0
        teacher_output = None
        if teacher is not None:
            with torch.no_grad(), autocast("cuda", enabled=scaler.is_enabled()):
                teacher_output = teacher(
                    input_ids=input_ids,
                    labels=input_ids,
                    output_hidden_states=output_hidden,
                )

        if bool(getattr(train_cfg, "gradient_checkpointing", False)):
            student.train()
        else:
            student.eval()
        for record in records:
            record.replacement.train()
        with autocast("cuda", enabled=scaler.is_enabled()):
            student_output = student(
                input_ids=input_ids,
                labels=input_ids,
                output_hidden_states=output_hidden,
            )
            loss, components = _joint_lm_distillation_loss(
                student_output,
                teacher_output,
                hidden_layers=hidden_layers,
                lm_loss_weight=float(train_cfg.lm_loss_weight),
                kl_loss_weight=float(train_cfg.kl_loss_weight),
                kl_temperature=float(train_cfg.kl_temperature),
                hidden_loss_weight=float(train_cfg.hidden_loss_weight),
            )

        scaler.scale(loss).backward()
        max_grad_norm = getattr(train_cfg, "max_grad_norm", None)
        if max_grad_norm is not None and float(max_grad_norm) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, float(max_grad_norm))
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(float(loss.detach().float().item()))

        if step % int(train_cfg.eval_every) == 0 or step == int(train_cfg.max_steps):
            valid_metrics = _evaluate_lm_loss_on_tokens(
                student,
                valid_tokens,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                max_batches=eval_batches,
            )
            valid = float(valid_metrics["lm_loss"])
            valid_losses.append(valid)
            if valid < best_valid:
                best_valid = valid
                best_step = step
            for record in records:
                record.replacement.train()

        if step % int(train_cfg.log_every) == 0 or step == 1:
            logger.info(
                (
                    "joint LM distillation step=%s/%s total=%.6f lm=%.6f "
                    "kl=%.6f hidden=%.6f valid_lm=%.6f"
                ),
                step,
                train_cfg.max_steps,
                components["total"],
                components["lm"],
                components["kl"],
                components["hidden"],
                valid_losses[-1],
            )

    units = _records_to_units(records)
    result = TransformerReplacementTrainingResult(
        train_losses=train_losses,
        valid_losses=valid_losses,
        initial_valid_loss=initial_valid,
        final_valid_loss=valid_losses[-1],
        best_valid_loss=best_valid,
        best_step=best_step,
        save_dir=train_cfg.save_dir,
        layer_indices=[unit.layer_index for unit in units],
    )
    _save_result(result, units, save_replacements=train_cfg.save_replacements)
    return result


def run_transformer_replacement_training(
    config: Config,
) -> TransformerReplacementTrainingResult:
    """Run the configured transformer replacement distillation experiment."""
    train_cfg = config.training.transformer_replacement
    if not train_cfg.enabled:
        raise ValueError("training.transformer_replacement.enabled must be true")
    mode = str(train_cfg.mode).lower()
    if mode not in {"layerwise_distillation", "joint_lm_distillation"}:
        raise ValueError(
            "mode must be 'layerwise_distillation' or 'joint_lm_distillation'"
        )
    if (
        mode == "layerwise_distillation"
        and str(train_cfg.train_target).lower() != "replacement_only"
    ):
        raise ValueError(
            "Only train_target='replacement_only' is currently implemented"
        )

    seed = int(train_cfg.seed if train_cfg.seed is not None else config.experiment.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _resolve_device(train_cfg.device)
    dtype = _resolve_dtype(train_cfg.dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        logger.warning("Using float32 for CPU transformer replacement training")
        dtype = torch.float32

    source = str(train_cfg.teacher_source).lower()
    if mode == "joint_lm_distillation":
        if source != "hf_text":
            raise ValueError(
                "joint_lm_distillation currently requires teacher_source='hf_text'"
            )
        return _train_joint_hf_text_source(config, device=device, dtype=dtype)

    if source == "synthetic_mlp":
        units = _make_synthetic_units(config, device=device, dtype=dtype)
        train_datasets, valid_datasets = _build_synthetic_datasets(
            units,
            config,
            device=device,
            dtype=dtype,
        )
        return _train_tensor_dataset_source(
            units,
            train_datasets,
            valid_datasets,
            config,
            device=device,
            dtype=dtype,
        )
    if source == "hidden_cache":
        units, train_datasets, valid_datasets = _build_hidden_cache_units_and_datasets(
            config,
            device=device,
            dtype=dtype,
        )
        return _train_tensor_dataset_source(
            units,
            train_datasets,
            valid_datasets,
            config,
            device=device,
            dtype=dtype,
        )
    if source == "hf_text":
        return _train_hf_text_source(config, device=device, dtype=dtype)
    raise ValueError(
        "teacher_source must be 'synthetic_mlp', 'hidden_cache', or 'hf_text'"
    )


__all__ = [
    "DistillationUnit",
    "SyntheticTransformerMLP",
    "TransformerReplacementBenchmarkResult",
    "TransformerReplacementTrainingResult",
    "benchmark_saved_transformer_replacement",
    "run_transformer_replacement_training",
]
