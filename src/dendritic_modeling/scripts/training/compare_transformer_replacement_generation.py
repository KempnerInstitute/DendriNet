#!/usr/bin/env python
"""Compare original and dendritic-replaced causal-LM generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dendritic_modeling.config import load_config
from dendritic_modeling.networks.architectures.transformer import (
    apply_transformer_replacement_config,
)
from dendritic_modeling.training.transformer_replacement import _to_plain_mapping

DEFAULT_PROMPTS = [
    "Question: What is the main purpose of the hippocampus in memory?\nAnswer:",
    "Question: Explain why sparse neural connectivity can reduce model parameters.\nAnswer:",
    "Question: What is one advantage and one risk of replacing a transformer MLP with a dendritic layer?\nAnswer:",
    "Complete the sentence: In a biological neuron, dendrites are important because",
]


def _load_prompts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    prompt_path = Path(path)
    if prompt_path.suffix.lower() == ".json":
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
            return [str(item) for item in payload["prompts"]]
        raise ValueError("Prompt JSON must be a list or {'prompts': [...]}")
    return [
        block.strip()
        for block in prompt_path.read_text(encoding="utf-8").split("\n\n")
        if block.strip()
    ]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _first_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    outputs = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device=device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation_ids = generated[0, input_ids.shape[-1] :]
            outputs.append(
                {
                    "prompt": prompt,
                    "completion": tokenizer.decode(
                        continuation_ids,
                        skip_special_tokens=True,
                    ),
                    "full_text": tokenizer.decode(
                        generated[0],
                        skip_special_tokens=True,
                    ),
                }
            )
    return outputs


def _load_checkpoint_records(
    model: torch.nn.Module,
    config: Any,
    checkpoint_dir: Path,
) -> None:
    records = apply_transformer_replacement_config(
        model,
        config.model.transformer_replacement,
        config.model.core,
    )
    if not records:
        raise ValueError("No transformer replacement records were created")
    map_location = _first_device(model)
    for record in records:
        checkpoint_path = checkpoint_dir / f"layer_{record.layer_index}_replacement.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=map_location)
        state_dict = payload.get("state_dict", payload)
        record.replacement.load_state_dict(state_dict)


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Transformer Replacement Generation Comparison",
        "",
        f"- model: `{payload['model_name']}`",
        f"- checkpoint: `{payload['checkpoint_dir']}`",
        f"- max_new_tokens: `{payload['max_new_tokens']}`",
        "",
    ]
    for idx, item in enumerate(payload["comparisons"], start=1):
        lines.extend(
            [
                f"## Prompt {idx}",
                "",
                "```text",
                item["prompt"],
                "```",
                "",
                "### Original",
                "",
                "```text",
                item["original_completion"].strip(),
                "```",
                "",
                "### Dendritic Replacement",
                "",
                "```text",
                item["replacement_completion"].strip(),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", help="YAML config used for the replacement.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing layer_<idx>_replacement.pt files.",
    )
    parser.add_argument("--prompts", default=None, help="Prompt text or JSON file.")
    parser.add_argument("--output-json", default=None, help="Output JSON path.")
    parser.add_argument("--output-md", default=None, help="Output Markdown path.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_config.cache_clear()
    config = load_config(args.config_path)
    train_cfg = config.training.transformer_replacement
    checkpoint_dir = Path(args.checkpoint_dir or train_cfg.save_dir)
    output_json = Path(
        args.output_json or checkpoint_dir / "generation_comparison.json"
    )
    output_md = Path(args.output_md or checkpoint_dir / "generation_comparison.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for generation comparison") from exc

    model_cfg = config.model.transformer_replacement
    model_kwargs = _to_plain_mapping(model_cfg.model_kwargs)
    model_name = model_cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    device = _resolve_device(args.device)
    if "device_map" not in model_kwargs:
        model = model.to(device=device)
    else:
        device = _first_device(model)

    prompts = _load_prompts(args.prompts)
    original = _generate(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    _load_checkpoint_records(model, config, checkpoint_dir)
    replacement = _generate(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    comparisons = []
    for orig, repl in zip(original, replacement):
        comparisons.append(
            {
                "prompt": orig["prompt"],
                "original_completion": orig["completion"],
                "replacement_completion": repl["completion"],
                "original_full_text": orig["full_text"],
                "replacement_full_text": repl["full_text"],
            }
        )

    payload = {
        "model_name": model_name,
        "checkpoint_dir": str(checkpoint_dir),
        "max_new_tokens": int(args.max_new_tokens),
        "comparisons": comparisons,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, output_md)
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}))


if __name__ == "__main__":
    main()
