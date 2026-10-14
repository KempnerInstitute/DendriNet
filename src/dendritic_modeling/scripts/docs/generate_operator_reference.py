"""Generate the dendritic operator configuration reference from the schema.

Introspects the typed configuration dataclasses so the reference page can
never drift from the code. Regenerate with::

    python -m dendritic_modeling.scripts.docs.generate_operator_reference \
        docs/operator_reference.md
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from dendritic_modeling.config.model import (
    ArchitectureConfig,
    ConnectivityConfig,
    CoreConfig,
    MorphologyConfig,
    ReactivationConfig,
    SparsityConfig,
    SpatialConfig,
    TransferFunctionConfig,
)
from dendritic_modeling.utils.logging_config import setup_logging

_SECTIONS = [
    ("core", CoreConfig),
    ("core.architecture", ArchitectureConfig),
    ("core.connectivity", ConnectivityConfig),
    ("core.transfer", TransferFunctionConfig),
    ("core.morphology", MorphologyConfig),
    ("core.sparsity", SparsityConfig),
    ("core.reactivation", ReactivationConfig),
    ("core.spatial", SpatialConfig),
]


def _default_repr(field: dataclasses.Field) -> str:
    if field.default is not dataclasses.MISSING:
        return repr(field.default)
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            return repr(field.default_factory())  # type: ignore[misc]
        except Exception:
            return field.default_factory.__name__  # type: ignore[union-attr]
    return "(required)"


def render_reference() -> str:
    lines = [
        "# Dendritic operator configuration reference",
        "",
        "Auto-generated from the typed configuration dataclasses",
        "(`dendritic_modeling.config.model`). Do not edit by hand — regenerate",
        "with `python -m dendritic_modeling.scripts.docs.generate_operator_reference`.",
        "",
        "The seven operator axes: input domain (`transfer`, "
        "`spatial.input_transform`, `spatial.input_scale`), weight domain "
        "(`morphology.weight_transform`; `identity` = signed), integration "
        "rule (`morphology.use_shunting`, `morphology.additive_mode`), gate "
        "nonlinearity (`reactivation.type`, soma-stage override "
        "`reactivation.soma_type`), morphology (`architecture.*_branch_factors`, "
        "`morphology.somatic_synapses`), topology (`sparsity`, `connectivity`), "
        "and output contract (`spatial.output_adapter_mode`).",
        "",
    ]
    for path, cls in _SECTIONS:
        lines.append(f"## `{path}` ({cls.__name__})")
        doc = (cls.__doc__ or "").strip().splitlines()
        if doc:
            lines.append("")
            lines.append(doc[0])
        lines.append("")
        lines.append("| field | type | default |")
        lines.append("|---|---|---|")
        for field in dataclasses.fields(cls):
            type_name = getattr(field.type, "__name__", str(field.type))
            lines.append(
                f"| `{field.name}` | `{type_name}` | `{_default_repr(field)}` |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    output = Path(args[0]) if args else Path("docs/operator_reference.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_reference())
    print(f"Wrote {output}")


if __name__ == "__main__":
    setup_logging()
    main()
