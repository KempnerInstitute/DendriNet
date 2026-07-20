"""Profile dense and dendritic LM parameter scaling without loading weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dendritic_modeling.config import load_config
from dendritic_modeling.networks.architectures.transformer import (
    TransformerParameterSpec,
    profile_dendritic_lm_config,
    profile_transformer_component_options,
)


def _load_hf_config_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Hugging Face config JSON must contain an object")
    return payload


def _download_hf_config(model_name: str, *, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError(
            "transformers is required unless --hf-config-json is provided"
        ) from exc
    return AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=bool(trust_remote_code),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", help="Dendritic replacement YAML config")
    parser.add_argument(
        "--hf-config-json",
        default="",
        help="Local Hugging Face config.json (avoids all network access)",
    )
    parser.add_argument(
        "--ungated-mlp",
        action="store_true",
        help="Count a two-projection rather than gated three-projection FFN",
    )
    parser.add_argument(
        "--attention-layers",
        default="",
        help=(
            "Comma-separated attention layers or 'all'. When attention TopK "
            "values are set and this is empty, use the FFN replacement layers."
        ),
    )
    parser.add_argument("--attention-query-topk", type=int, default=None)
    parser.add_argument("--attention-key-value-topk", type=int, default=None)
    parser.add_argument("--attention-output-topk", type=int, default=None)
    parser.add_argument("--embedding-rank", type=int, default=None)
    parser.add_argument("--lm-head-rank", type=int, default=None)
    parser.add_argument("--output", default="", help="Optional JSON output path")
    return parser.parse_args()


def _parse_attention_layers(
    value: str,
    *,
    num_hidden_layers: int,
    default: list[int],
) -> list[int]:
    normalized = str(value).strip().lower()
    if not normalized:
        return list(default)
    if normalized == "all":
        return list(range(int(num_hidden_layers)))
    return [int(item.strip()) for item in normalized.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    load_config.cache_clear()
    experiment_config = load_config(args.config_path)
    replacement = experiment_config.model.transformer_replacement
    if args.hf_config_json:
        hf_config = _load_hf_config_json(args.hf_config_json)
    else:
        hf_config = _download_hf_config(
            replacement.model_name,
            trust_remote_code=bool(
                replacement.model_kwargs.get("trust_remote_code", False)
            ),
        )
    report = profile_dendritic_lm_config(
        experiment_config,
        hf_config,
        gated_mlp=not args.ungated_mlp,
    )
    component_requested = any(
        value is not None
        for value in (
            args.attention_query_topk,
            args.attention_key_value_topk,
            args.attention_output_topk,
            args.embedding_rank,
            args.lm_head_rank,
        )
    ) or bool(args.attention_layers)
    if component_requested:
        transformer = TransformerParameterSpec.from_hf_config(
            hf_config,
            gated_mlp=not args.ungated_mlp,
        )
        attention_layers = _parse_attention_layers(
            args.attention_layers,
            num_hidden_layers=transformer.num_hidden_layers,
            default=list(replacement.layers),
        )
        report["component_options"] = profile_transformer_component_options(
            transformer,
            attention_layers=attention_layers,
            query_topk=args.attention_query_topk,
            key_value_topk=args.attention_key_value_topk,
            output_topk=args.attention_output_topk,
            embedding_rank=args.embedding_rank,
            lm_head_rank=args.lm_head_rank,
        )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="", flush=True)


if __name__ == "__main__":
    main()
