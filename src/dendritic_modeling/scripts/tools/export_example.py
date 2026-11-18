"""List or export a small installed DendriNet example."""

from __future__ import annotations

import argparse
from pathlib import Path

from dendritic_modeling.examples import example_names, example_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", choices=example_names())
    parser.add_argument("--output", type=Path, help="Write a new YAML file.")
    args = parser.parse_args()
    if args.name is None:
        if args.output:
            parser.error("--output requires an example name")
        print("\n".join(example_names()))
        return
    text = example_text(args.name)
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        except OSError as error:
            parser.error(str(error))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
