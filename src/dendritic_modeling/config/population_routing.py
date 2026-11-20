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


def projection_delays(
    timing: str,
    delay_steps: int | None,
    delay_steps_by_level: list[int] | None,
    n_levels: int | None = None,
) -> tuple[int, ...]:
    """Validate travel times and return update counts in distal-to-soma order."""
    if delay_steps_by_level is not None:
        if delay_steps is not None or timing != "delayed":
            raise ValueError(
                "delay_steps_by_level requires delayed timing and no delay_steps"
            )
        if not isinstance(delay_steps_by_level, list) or not delay_steps_by_level:
            raise ValueError(
                "delay_steps_by_level must be a nonempty list of positive integers"
            )
        if any(type(step) is not int or step < 1 for step in delay_steps_by_level):
            raise ValueError("delay_steps_by_level entries must be positive integers")
        if n_levels is not None and len(delay_steps_by_level) != n_levels:
            raise ValueError(
                f"delay_steps_by_level needs {n_levels} values, distal to soma"
            )
        return tuple(delay_steps_by_level)
    steps = int(timing == "delayed") if delay_steps is None else delay_steps
    if type(steps) is not int:
        raise ValueError("delay_steps must be an integer number of updates")
    if timing == "delayed" and steps < 1:
        raise ValueError("delay_steps must be positive for delayed connections")
    if timing == "same_step" and steps != 0:
        raise ValueError("same_step connections require zero delay_steps")
    return (steps,) * (n_levels if n_levels is not None else 1)
