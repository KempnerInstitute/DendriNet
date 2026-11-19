"""Export the empirical reader examples from committed, explicitly listed files."""

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

# Kept here so a development file cannot enter the bundle through a glob.
FILES = (
    "README.md",
    "analysis/calibrate_context.py",
    "analysis/plot_software_guide.py",
    "analysis/reproduce.py",
    "configs/cifar_primary_additive.yaml",
    "configs/cifar_primary_shunting.yaml",
    "configs/cifar_three_layer_additive.yaml",
    "configs/cifar_three_layer_shunting.yaml",
    "configs/context_additive.yaml",
    "configs/context_shunting.yaml",
    "configs/mnist_additive.yaml",
    "configs/mnist_shunting.yaml",
    "reproducibility/configs.json",
    "reproducibility/export_examples.py",
    "reproducibility/figures.md",
    "reproducibility/source_data.json",
)
PROJECT = "projects/empirical/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Reviewed study source-data ZIP")
    parser.add_argument("--output", type=Path, required=True, help="New reader ZIP")
    parser.add_argument("--revision", default="HEAD", help="DendriNet Git revision")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]

    def git(*words):
        return subprocess.check_output(["git", "-C", str(root), *words])

    revision = (
        git("rev-parse", "--verify", args.revision + "^{commit}").decode().strip()
    )
    tree = git("ls-tree", "-r", "-z", revision, "--", PROJECT)
    entries = {}
    for record in tree.rstrip(b"\0").split(b"\0"):
        metadata, name = record.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        if mode != b"100644" or kind != b"blob":
            raise ValueError(f"Unexpected study file type: {name!r}")
        entries[name.decode()[len(PROJECT) :]] = oid.decode()
    if set(entries) != set(FILES):
        raise ValueError(
            f"Study inventory differs from the public list: {sorted(set(entries) ^ set(FILES))}"
        )
    assets = {name: git("cat-file", "blob", oid) for name, oid in entries.items()}
    manifest = json.loads(assets["reproducibility/source_data.json"])
    data = args.data.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise ValueError("Source-data archive differs from the reviewed asset")
    with zipfile.ZipFile(io.BytesIO(data)) as inner:
        if sorted(inner.namelist()) != sorted(manifest["members"]):
            raise ValueError("Unexpected numerical table inventory")
        for name, expected in manifest["members"].items():
            if hashlib.sha256(inner.read(name)).hexdigest() != expected:
                raise ValueError(f"Changed table: {name}")
    assets[manifest["filename"]] = data
    assets["SOURCE_COMMIT.txt"] = (revision + "\n").encode()
    with zipfile.ZipFile(args.output, "x") as archive:
        for name, content in sorted(assets.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED)


if __name__ == "__main__":
    main()
