# Jobs

Cluster job definitions — Slurm batch/array scripts (`.sbatch` / `.slurm`) and
the experiment-specific configs they consume. These **launch and run** the
package on a scheduler; they are not part of the importable library.

## `jobs/` vs. `src/dendritic_modeling/scripts/`

|                              | `jobs/`                                              | `src/dendritic_modeling/scripts/`                          |
| ---------------------------- | --------------------------------------------------- | ---------------------------------------------------------- |
| Contents                     | Slurm `.sbatch`/`.slurm` launchers + run configs    | Python CLI entry points                                    |
| Role                         | Schedule/run a specific experiment or benchmark     | Reusable, installed commands (`dendritic-train`, `dendritic-sweep`, …) |
| Installed with the package?  | No                                                  | Yes (via `pip install -e .`)                               |
| Stability                    | Experiment-specific; may be ad hoc                  | Maintained package surface                                 |

Rule of thumb: **reusable Python logic belongs in the package**
(`dendritic_modeling.*`, with thin CLIs under `scripts/`); **a way to launch a
specific run or campaign on a cluster belongs here.**

## Contents

- `profile_nsys.sbatch`, `profile_ncu.sbatch` — Nsight profiling captures (see
  [docs/profiling/](../docs/profiling/)).
- `transformer_replacement_benchmarks/` — Slurm scripts and configs for the
  transformer-replacement benchmark campaign (see its
  [README](transformer_replacement_benchmarks/README.md)).
