#!/bin/bash
# KEMPNER INSIGHT SHOWCASE: the kernel improvement, visible on the org's own
# job dashboards. Submits TWO jobs at the same moment, identical in every
# dashboard dimension -- same partition, same 1x H100, same CPUs/memory, same
# dataset (pre-cached, no download valley), same 50 epochs, same seed, the
# EXACT frozen configs of the completed comparison runs. The only difference
# Insight can show is the one that matters:
#
#     DENDRINET-ff-BASELINE-50ep   ~77 min elapsed
#     DENDRINET-ff-OPTIMIZED-50ep  ~26 min elapsed   <- 3x, same work
#
# Both charts read ~continuous GPU utilization; they start together and the
# optimized one simply FINISHES 50 minutes earlier. Duration at equal work is
# the honest dashboard metric (utilization % cannot distinguish efficient
# from wasteful busy-ness).
set -euo pipefail
cd /n/home12/rwhalen/DendriNet

# no-download precondition: the demo must not include network noise
[ -d data/datasets/cifar/cifar-10-batches-py ] || {
  echo "CIFAR not staged at data/datasets/cifar -- aborting (the valley would poison the demo)"; exit 1; }

# copy the frozen configs with results redirected -- a raw rerun would
# OVERWRITE the original runs' results dirs (results_dir is baked in)
mkdir -p outputs/insight_showcase
sed 's|results_dir: .*|results_dir: outputs/insight_showcase/baseline|' \
  outputs/indexed_backends/feedforward_no_opt/ff_large_no_opt_20260811235440/configs/unified_config_0.yaml \
  > outputs/insight_showcase/baseline_config.yaml
sed 's|results_dir: .*|results_dir: outputs/insight_showcase/optimized|' \
  outputs/indexed_backends/feedforward_optimized/ff_large_optimized_20260812024650/configs/unified_config_0.yaml \
  > outputs/insight_showcase/optimized_config.yaml
# the harness derives its dataset cache from the results root (that is how
# the first optimized run re-downloaded despite a repo-level cache, and how
# showcase attempt 1 re-downloaded twice more). Pre-seed BOTH derived cache
# locations with symlinks to the extracted data -- knob-independent.
for arm in baseline optimized; do
  mkdir -p "outputs/insight_showcase/$arm/datasets/cifar"
  ln -sfn /n/home12/rwhalen/DendriNet/data/cifar/cifar-10-batches-py \
    "outputs/insight_showcase/$arm/datasets/cifar/cifar-10-batches-py"
done

BASE_CFG=outputs/insight_showcase/baseline_config.yaml
OPT_CFG=outputs/insight_showcase/optimized_config.yaml
PY=/n/home12/rwhalen/.conda/envs/dendritic_env/bin/python
COMMON=(--account=kempner_undergrads --partition=kempner_h100_priority
        --chdir=/n/home12/rwhalen/DendriNet --nodes=1 --ntasks-per-node=1
        --gres=gpu:1 --cpus-per-task=8 --mem=64GB --time=02:00:00)

sbatch "${COMMON[@]}" --job-name=DENDRINET-ff-BASELINE-50ep \
  --output=outputs/insight_showcase_baseline_%j.log \
  --wrap "export PYTHONPATH=\$(pwd)/src; $PY src/dendritic_modeling/scripts/training/train_experiments.py $BASE_CFG"

sbatch "${COMMON[@]}" --job-name=DENDRINET-ff-OPTIMIZED-50ep \
  --output=outputs/insight_showcase_optimized_%j.log \
  --wrap "export PYTHONPATH=\$(pwd)/src; $PY src/dendritic_modeling/scripts/training/train_experiments.py $OPT_CFG"

echo "Both submitted. Watch them side by side in Kempner Insight:"
echo "  same start, same resources, same work -- one ends ~50 minutes early."
