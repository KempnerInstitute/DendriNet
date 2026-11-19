"""Small, installed examples for tutorials and graphical clients.

These are demonstration settings, not the qualified recipes of any paper.
"""

from importlib.resources import files


def example_names() -> tuple[str, ...]:
    """Return the names of the installed YAML examples."""
    return tuple(
        sorted(
            p.name[:-5]
            for p in files(__package__).iterdir()
            if p.name.endswith(".yaml")
        )
    )


def example_text(name: str) -> str:
    """Read an installed example by its exact name, without writing files."""
    if name not in example_names():
        raise ValueError(f"Unknown example {name!r}; choose from {example_names()}")
    return files(__package__).joinpath(f"{name}.yaml").read_text(encoding="utf-8")
