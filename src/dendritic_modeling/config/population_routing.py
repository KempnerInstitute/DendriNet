"""Population routing conventions shared by validation, construction and inspection."""

INPUT_POLARITIES = {
    "input": "excitatory",
    "input_e": "excitatory",
    "input_i": "inhibitory",
}
INPUT_ALIASES = {
    "input": "input",
    "input_e": "input_e",
    "input_exc": "input_e",
    "input_excitatory": "input_e",
    "input_i": "input_i",
    "input_inh": "input_i",
    "input_inhibitory": "input_i",
}


def canonical_input_source(source: str) -> str | None:
    return INPUT_ALIASES.get(str(source).lower())


def population_readout(
    polarities: dict[str, str], configured: str | None
) -> str | None:
    """The first excitatory population supplies the default layer readout."""
    return configured or next(
        (name for name, sign in polarities.items() if sign == "excitatory"), None
    )


def projection_pathway(timing: str, source_polarity: str, explicit: str) -> str:
    """An explicit pathway takes precedence over source polarity."""
    prefix = "ff" if timing == "same_step" else "rec"
    return explicit or f"{prefix}_{source_polarity}"
