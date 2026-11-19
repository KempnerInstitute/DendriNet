"""Recompute the published seed summaries for Figure 3D and Figure 7F on a CPU."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from dendritic_modeling.analysis.statistics import bootstrap_mean_interval

PROJECT = Path(__file__).resolve().parents[1]
ARMS = ("main", "additive", "matched_additive", "bernardo")
SITES = ("intact", "L1 I +1", "L2 I +1")


def read_source_tables(archive):
    manifest = json.loads((PROJECT / "reproducibility/source_data.json").read_text())
    data = archive.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise ValueError("The source-data archive differs from the study manifest")
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        if sorted(bundle.namelist()) != sorted(manifest["members"]):
            raise ValueError("Unexpected source table inventory")
        return {key: bundle.read(key) for key in manifest["members"]}


def transmission_summary(data):
    # The original summarizer operated on full-precision measurements before writing CSV.
    frame = pd.read_csv(io.BytesIO(data), float_precision="round_trip")
    rng = np.random.default_rng(20260924)
    rows = []
    for (rule, depth), group in frame.groupby(["rule", "depth"], sort=True):
        group = group.sort_values("seed")
        if list(group.seed) != list(range(20)):
            raise ValueError(f"Unexpected seed membership for {rule}, depth {depth}")
        for metric in ("mean", "rms", "fraction_below_1e_6"):
            values = group[metric].to_numpy()
            samples = rng.integers(0, len(values), (10000, len(values)))
            lower, upper = np.quantile(values[samples].mean(axis=1), [0.025, 0.975])
            rows.append((rule, depth, metric, values.mean(), lower, upper, len(values)))
    return pd.DataFrame(
        rows, columns=["rule", "depth", "metric", "mean", "lower", "upper", "n_units"]
    )


def context_summary(data):
    # The figure reads the stored fractions with pandas' default CSV parser.
    frame = pd.read_csv(io.BytesIO(data))
    rows = []
    for arm in ARMS:
        for site in SITES:
            group = frame[(frame.arm == arm) & (frame.intervention == site)]
            if len(group) != 10 or group.seed.nunique() != 10:
                raise ValueError(
                    f"Expected ten independently trained networks for {arm}, {site}"
                )
            for metric in ("accuracy", "donor_label"):
                estimate = bootstrap_mean_interval(
                    100 * group[metric].to_numpy(), draws=10000, seed=20261021
                )
                rows.append(
                    (
                        arm,
                        site,
                        metric,
                        estimate.mean,
                        estimate.lower,
                        estimate.upper,
                        len(group),
                    )
                )
    return pd.DataFrame(
        rows,
        columns=["arm", "intervention", "metric", "mean", "lower", "upper", "n_units"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    args = parser.parse_args()
    tables = read_source_tables(args.archive)
    transmission = transmission_summary(tables["transmission/seed_units.csv"])
    result = transmission.to_csv(index=False).encode()
    if result != tables["transmission/summary.csv"]:
        raise ValueError(
            "Transmission replay differs from the published table. Check NumPy and pandas versions."
        )
    context = context_summary(tables["context/routing.csv"])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.csv").write_bytes(result)
    context.to_csv(args.output / "context_summary.csv", index=False)


if __name__ == "__main__":
    main()
