"""
Generate ablation config variants: conv1_only .. fc7_only, early_only, late_only, all_swapped.
fc8 is always standard. Run each with >=3 seeds.
"""

from typing import List, Tuple

# conv_type: 5 elements for conv1..conv5; fc_type: 2 elements for fc6, fc7
ABLATION_VARIANTS: List[Tuple[str, List[str], List[str]]] = [
    ("conv1_only", ["dendritic", "standard", "standard", "standard", "standard"], ["standard", "standard"]),
    ("conv2_only", ["standard", "dendritic", "standard", "standard", "standard"], ["standard", "standard"]),
    ("conv3_only", ["standard", "standard", "dendritic", "standard", "standard"], ["standard", "standard"]),
    ("conv4_only", ["standard", "standard", "standard", "dendritic", "standard"], ["standard", "standard"]),
    ("conv5_only", ["standard", "standard", "standard", "standard", "dendritic"], ["standard", "standard"]),
    ("fc6_only", ["standard"] * 5, ["dendritic", "standard"]),
    ("fc7_only", ["standard"] * 5, ["standard", "dendritic"]),
    ("early_only", ["dendritic", "dendritic", "dendritic", "standard", "standard"], ["standard", "standard"]),
    ("late_only", ["standard", "standard", "standard", "dendritic", "dendritic"], ["dendritic", "dendritic"]),
    ("all_swapped", ["dendritic"] * 5, ["dendritic", "dendritic"]),
]

# Hybrid encoder output dim (FC7_DIM)
HYBRID_ENCODER_OUTPUT_DIM = 4096


def get_conv_type_fc_type(variant_name: str) -> Tuple[List[str], List[str]]:
    """Return (conv_type, fc_type) for a named variant."""
    for name, conv_type, fc_type in ABLATION_VARIANTS:
        if name == variant_name:
            return conv_type, fc_type
    raise ValueError(f"Unknown variant: {variant_name}")
