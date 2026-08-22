Frozen code snapshot — 3-seed sweep (taken 2026-08-12 16:02, transform patched 16:40)
=====================================================================================

This is the EXACT code the 3-seed sweep executes from. The SLURM workers set
PYTHONPATH to this tree's src/, not to the working tree, so this directory —
not the branch checkout — is what produced the seeded results.

It carries changes that are NOT committed to the branch itself, which is why a
checkout of the branch alone cannot reproduce the sweep:

  networks/architectures/factory.py        +122 lines   point / sparse-point cores
  scripts/script_utils/setup_utils.py       +66 lines
  config/model.py                            +3 lines

UNCOMMITTED.patch holds those diffs against GIT_HEAD.txt; GIT_STATUS.txt records
the tree state at snapshot time.

BOTH preprocessing regimes are present. This is the one difference that separates
the two sets of results, and which of them the paper uses is not yet decided:

  datasets/standard_datasets.py        Resize(256) + CenterCrop(224)
      -> the 3-seed sweep (seeds 41/42/43, N_E 32..4096), run dirs named
         *_seed4[123]_20260[89]*

  datasets/standard_datasets.py.orig   Resize((224, 224))   [squash]
      -> the older single-seed runs, and every array under
         outputs/interpretability_results/

The two differ by roughly 1.4 points top-1, uniformly across models, so results
from the two regimes must not be pooled or compared directly. To run under the
old regime, copy the .orig file over standard_datasets.py.
