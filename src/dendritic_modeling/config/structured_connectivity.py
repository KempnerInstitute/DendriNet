"""Keys, validation and resolution of structured-connectivity blocks.

A structured-connectivity block (``model.core.connectivity.structured`` or a
population's ``structured_connectivity``) has root defaults and per-pathway
entries. Every key a mask, index or spatial sampler reads is listed here, so a
misspelled key, pathway name or an enabled block without any rule raises
instead of silently leaving the pathway dense.

Four rule keys have a second spelling (:data:`RULE_KEY_ALIASES`). The root and
every pathway entry are rewritten to the canonical names before the pathway
entry is merged over the root defaults, so the spelling never decides which
value applies. This matters for typed blocks, whose root always carries
``probability``, ``indegree``, ``distance_kernel`` and ``distance_sigma``. A
rule that writes both spellings of one key, and a ``pathways`` mapping that
names one pathway twice, raise.
"""

from __future__ import annotations

from typing import Any

from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.keys import check_section_keys

#: Public pathway names and their canonical E/I pathway.
PATHWAY_ALIASES = {
    "ee": "ee",
    "e_to_e": "ee",
    "exc_to_exc": "ee",
    "excitatory_to_excitatory": "ee",
    "ie": "ie",
    "i_to_e": "ie",
    "inh_to_exc": "ie",
    "inhibitory_to_excitatory": "ie",
    "ei": "ei",
    "e_to_i": "ei",
    "exc_to_inh": "ei",
    "excitatory_to_inhibitory": "ei",
    "ii": "ii",
    "i_to_i": "ii",
    "inh_to_inh": "ii",
    "inhibitory_to_inhibitory": "ii",
    "rec_ee": "rec_ee",
    "recurrent_ee": "rec_ee",
    "rec_e_to_e": "rec_ee",
    "rec_ie": "rec_ie",
    "recurrent_ie": "rec_ie",
    "rec_i_to_e": "rec_ie",
    "rec_ei": "rec_ei",
    "recurrent_ei": "rec_ei",
    "rec_e_to_i": "rec_ei",
    "rec_ii": "rec_ii",
    "recurrent_ii": "rec_ii",
    "rec_i_to_i": "rec_ii",
}

#: Root or pathway keys whose value defines a mask rule.
MASK_RULE_KEYS = (
    "probability",
    "p_connect",
    "indegree",
    "in_degree",
    "pre_positions",
    "post_positions",
    "ranges_by_level",
    "inventory_counts",
    "feature_ranges",
)

#: Keys of one pathway rule (also allowed as root defaults).
RULE_KEYS = frozenset(
    {
        *MASK_RULE_KEYS,
        "method",
        "seed",
        "distance_kernel",
        "kernel",
        "distance_sigma",
        "sigma",
        "base_probability",
        "tier_groups",
        "owner_count",
        "manifest_path",
        "spatial",
    }
)
ROOT_KEYS = RULE_KEYS | {"enabled", "pathways"}
#: Second spellings of rule keys and the canonical key each one stands for.
RULE_KEY_ALIASES = {
    "p_connect": "probability",
    "in_degree": "indegree",
    "kernel": "distance_kernel",
    "sigma": "distance_sigma",
}
#: Methods that keep a pathway dense on purpose.
DENSE_METHODS = frozenset({"none", "dense", "all"})
#: Names of the distance-dependent mask method.
DISTANCE_METHODS = frozenset({"distance", "distance_kernel"})
#: ``seed`` of a population rule whose random masks (``bernoulli``,
#: ``fixed_indegree``, ``distance``) draw from the global random stream when
#: the model is built, as unseeded rules did in config_version 2; training
#: does not replace it with a seed.
GLOBAL_RNG_SEED = "global_rng"
SPATIAL_KEYS = frozenset(
    {"input_shape", "split_axes", "region_mode", "minimum_window_shape", "region_seed"}
)


def normalize_pathway_name(pathway: str) -> str:
    """Return the canonical E/I pathway of a public pathway name."""
    normalized = str(pathway).lower()
    return PATHWAY_ALIASES.get(normalized, normalized)


def _has_rule(rule: dict[str, Any]) -> bool:
    return any(rule.get(key) is not None for key in MASK_RULE_KEYS) or bool(
        rule.get("manifest_path") or to_plain_dict(rule.get("spatial") or {})
    )


def canonical_rule(rule: Any, *, section: str) -> dict[str, Any]:
    """Return ``rule`` with alias keys renamed to their canonical names.

    Raises when the rule writes both spellings of one key, since one of the
    two values would be ignored.
    """
    rule = to_plain_dict(rule or {})
    both = [
        f"{canonical!r} and its alias {alias!r} (keep {canonical!r})"
        for alias, canonical in RULE_KEY_ALIASES.items()
        if alias in rule and canonical in rule
    ]
    if both:
        raise ValueError(
            f"{section} sets both spellings of one key, so one value would be "
            "ignored: " + "; ".join(both)
        )
    return {RULE_KEY_ALIASES.get(key, key): value for key, value in rule.items()}


def _check_seed(rule: dict[str, Any], *, section: str) -> None:
    seed = rule.get("seed")
    if seed is None or seed == GLOBAL_RNG_SEED:
        return
    try:
        if isinstance(seed, bool):
            raise TypeError
        int(seed)
    except (TypeError, ValueError):
        raise ValueError(
            f"{section}.seed={seed!r} must be an integer, null, or "
            f"{GLOBAL_RNG_SEED!r} (masks from the global random stream)"
        ) from None


def _check_rule(rule: dict[str, Any], *, allowed, section: str) -> None:
    check_section_keys(rule, allowed, section=section)
    canonical_rule(rule, section=section)
    _check_seed(rule, section=section)
    spatial = to_plain_dict(rule.get("spatial") or {})
    check_section_keys(spatial, SPATIAL_KEYS, section=f"{section}.spatial")


def validate_structured_connectivity(config: Any, *, section: str) -> None:
    """Reject unknown keys and pathway names, and an enabled block with no rule.

    Values are only checked, never changed. A rule is a mask parameter
    (``probability``, ``indegree``, positions, feature ranges or inventory),
    a ``spatial`` block or an index ``manifest_path``, at the root or in a
    pathway entry; ``method: dense``/``none`` entries keep a pathway dense on
    purpose. In an enabled block every ``distance`` rule needs a connection
    probability at distance 0 (see :func:`distance_base_probability`).
    """
    root = to_plain_dict(config or {})
    if not root:
        return
    _check_rule(root, allowed=ROOT_KEYS, section=section)
    pathways = to_plain_dict(root.get("pathways") or {})
    unknown = sorted(
        name for name in pathways if str(name).lower() not in PATHWAY_ALIASES
    )
    if unknown:
        raise ValueError(
            f"{section}.pathways has unknown pathway name(s) {unknown}; use "
            f"{sorted(set(PATHWAY_ALIASES.values()))} or their aliases"
        )
    spellings: dict[str, list[str]] = {}
    for name in pathways:
        spellings.setdefault(normalize_pathway_name(name), []).append(str(name))
    repeated = {
        canonical: names for canonical, names in spellings.items() if len(names) > 1
    }
    if repeated:
        raise ValueError(
            f"{section}.pathways names one pathway more than once ("
            + "; ".join(
                f"{', '.join(names)} are all {canonical!r}"
                for canonical, names in sorted(repeated.items())
            )
            + "), and only one entry would apply; keep one entry per pathway"
        )
    explicit_dense = False
    for name, entry in pathways.items():
        entry = to_plain_dict(entry or {})
        _check_rule(entry, allowed=RULE_KEYS, section=f"{section}.pathways.{name}")
        explicit_dense = (
            explicit_dense or str(entry.get("method", "")).lower() in DENSE_METHODS
        )
    if not bool(root.get("enabled", False)):
        return
    if (
        not _has_rule(root)
        and not explicit_dense
        and not any(
            _has_rule(to_plain_dict(entry or {})) for entry in pathways.values()
        )
    ):
        raise ValueError(
            f"{section} is enabled but defines no rule (probability, indegree, "
            "positions, feature ranges, spatial or manifest_path at the root or "
            "in a pathway), so every pathway would stay dense"
        )
    for where, rule in _pathway_rules(root, section=section):
        if str(rule.get("method", "bernoulli")).lower() in DISTANCE_METHODS:
            distance_base_probability(rule, section=where)


def _root_defaults(root: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in root.items() if key not in {"pathways", "enabled"}
    }


def _root_has_rule(root_defaults: dict[str, Any]) -> bool:
    """Whether the root alone is a rule for pathways without an entry.

    A typed block carries defaults such as ``method: bernoulli`` and an empty
    ``spatial`` mapping, which alone are not a rule, and a root
    ``manifest_path`` is a default for the pathway entries.
    """
    return any(root_defaults.get(key) is not None for key in MASK_RULE_KEYS) or bool(
        to_plain_dict(root_defaults.get("spatial") or {})
    )


def _pathway_rules(
    root: dict[str, Any], *, section: str
) -> list[tuple[str, dict[str, Any]]]:
    """Every rule a pathway of the block can follow, with where it is written.

    The root alone (for pathways without an entry, when it is a rule) and each
    pathway entry merged over the root, all with canonical key names.
    """
    root = canonical_rule(root, section=section)
    defaults = _root_defaults(root)
    rules = [(section, defaults)] if _root_has_rule(defaults) else []
    for name, entry in to_plain_dict(root.get("pathways") or {}).items():
        where = f"{section}.pathways.{name}"
        entry = canonical_rule(entry, section=where)
        if entry:
            rules.append((where, {**defaults, **entry}))
    return rules


def distance_base_probability(rule: dict[str, Any], *, section: str) -> float:
    """The connection probability at distance 0 of a ``distance`` rule.

    An omitted ``base_probability`` takes the rule's ``probability``, and
    ``1.0`` when the rule has neither (a population rule; the typed root of
    ``model.core.connectivity.structured`` always has ``probability``, which
    is null unless written). A null or non-numeric value raises.
    """
    key = "base_probability" if "base_probability" in rule else "probability"
    value = rule.get(key, 1.0)
    if value is None:
        if key == "probability":
            raise ValueError(
                f"{section} uses method 'distance' without base_probability, the "
                "connection probability at distance 0; write base_probability in "
                "the rule (an omitted base_probability takes the rule's "
                "probability, which is not set)"
            )
        raise ValueError(
            f"{section}.base_probability is null; method 'distance' needs the "
            "connection probability at distance 0, a number in [0, 1]"
        )
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{section}.{key}={value!r} is the base_probability of method "
            "'distance' and must be one number in [0, 1]"
        ) from None


def resolve_pathway_rule(
    config: Any, pathway: str, *, section: str = "structured connectivity"
) -> dict[str, Any]:
    """Return the rule of one pathway: the root defaults, overridden by its entry.

    The block is validated first. Alias keys of the root and of the pathway
    entry are renamed to their canonical names before the entry is merged over
    the root. Returns ``{}`` (a dense pathway) when the block is disabled, or
    when the pathway has no entry and the root alone is not a rule (it has no
    mask parameter and no ``spatial`` block).
    """
    root = to_plain_dict(config or {})
    validate_structured_connectivity(root, section=section)
    if not bool(root.get("enabled", False)):
        return {}
    root = canonical_rule(root, section=section)
    root_defaults = _root_defaults(root)
    canonical_pathway = normalize_pathway_name(pathway)
    pathway_rule: dict[str, Any] = {}
    for name, entry in to_plain_dict(root.get("pathways") or {}).items():
        if normalize_pathway_name(name) == canonical_pathway:
            pathway_rule = canonical_rule(entry, section=f"{section}.pathways.{name}")
            break
    if not pathway_rule and not _root_has_rule(root_defaults):
        return {}
    return {**root_defaults, **pathway_rule}


__all__ = [
    "DENSE_METHODS",
    "DISTANCE_METHODS",
    "GLOBAL_RNG_SEED",
    "MASK_RULE_KEYS",
    "PATHWAY_ALIASES",
    "ROOT_KEYS",
    "RULE_KEYS",
    "RULE_KEY_ALIASES",
    "SPATIAL_KEYS",
    "canonical_rule",
    "distance_base_probability",
    "normalize_pathway_name",
    "resolve_pathway_rule",
    "validate_structured_connectivity",
]
