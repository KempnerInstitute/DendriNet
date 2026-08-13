#!/bin/bash
# KEMPNER INSIGHT SHOWCASE, THREE-ARM: the full gather-kernel ladder on the
# org's own dashboards. Submits THREE jobs at the same moment, identical in
# every dashboard dimension -- same partition, same 1x H100, same 8 CPU/64G,
# same pre-cached dataset (NO download valley), same 50 epochs, same seed,
# the same frozen feedforward_large recipe. The ONLY difference is the
# gather backend, requested EXPLICITLY per arm (no auto-dispatch ambiguity):
#
#     DENDRINET-ff-EAGER-50ep     control_eager   ~76 min   (measured 1:15:50)
#     DENDRINET-ff-FUSED-50ep     ryan_fused      ~25 min   (measured 25:02)
#     DENDRINET-ff-TRITONV3-50ep  ryan_triton      ~9 min   (measured  9:10*)
#                                    *at 16CPU/128G; this run is the matched one
#
# All three start together; Insight shows them dropping off the running list
# one by one at equal work. Duration at fixed work is the honest metric.
set -euo pipefail
cd /n/home12/rwhalen/DendriNet

# no-download precondition: check the EXTRACTED payload the symlinks target
[ -d data/cifar/cifar-10-batches-py ] || {
  echo "extracted CIFAR missing at data/cifar/cifar-10-batches-py -- aborting"; exit 1; }

SHOW=outputs/insight_3way
mkdir -p "$SHOW"

# Frozen configs of the completed runs, with results redirected (a raw rerun
# would OVERWRITE the originals' results dirs -- results_dir is baked in).
SRC_EAGER=outputs/indexed_backends/feedforward_no_opt/ff_large_no_opt_20260811235440/configs/unified_config_0.yaml
SRC_FUSED=outputs/indexed_backends/feedforward_optimized/ff_large_optimized_20260812024650/configs/unified_config_0.yaml
SRC_TRITON=outputs/indexed_backends/feedforward_triton/ff_large_triton_20260812133633/configs/unified_config_0.yaml

sed 's|results_dir: .*|results_dir: outputs/insight_3way/eager|' \
  "$SRC_EAGER" > "$SHOW/eager_config.yaml"
# fused arm: request fused EXPLICITLY (the source config said ryan_auto, which
# happened to resolve to fused; explicit beats resolved for an A/B/C receipt)
sed -e 's|results_dir: .*|results_dir: outputs/insight_3way/fused|' \
    -e 's|comparison_backend: ryan_auto|comparison_backend: ryan_fused|' \
    -e 's|_sweep_variant: ryan_auto|_sweep_variant: ryan_fused|' \
  "$SRC_FUSED" > "$SHOW/fused_config.yaml"
sed 's|results_dir: .*|results_dir: outputs/insight_3way/triton|' \
  "$SRC_TRITON" > "$SHOW/triton_config.yaml"

grep -q "comparison_backend: ryan_fused" "$SHOW/fused_config.yaml" || {
  echo "fused sed did not take -- aborting"; exit 1; }

# The harness derives its dataset cache from each run's results root (this is
# how every fresh results dir re-downloaded, ~40 GPU-idle minutes each time).
# Pre-seed the derived cache location of ALL arms with symlinks -- verified
# against the actual torchvision loader path in the clean showcase pair.
for arm in eager fused triton; do
  mkdir -p "$SHOW/$arm/datasets/cifar"
  ln -sfn /n/home12/rwhalen/DendriNet/data/cifar/cifar-10-batches-py \
    "$SHOW/$arm/datasets/cifar/cifar-10-batches-py"
done

PY=/n/home12/rwhalen/.conda/envs/dendritic_env/bin/python
COMMON=(--account=kempner_undergrads --partition=kempner_h100_priority
        --chdir=/n/home12/rwhalen/DendriNet --nodes=1 --ntasks-per-node=1
        --gres=gpu:1 --cpus-per-task=8 --mem=64GB --time=02:00:00)

for arm_name in EAGER:eager FUSED:fused TRITONV3:triton; do
  NAME=${arm_name%%:*}; arm=${arm_name##*:}
  sbatch "${COMMON[@]}" --job-name="DENDRINET-ff-${NAME}-50ep" \
    --output="$SHOW/${arm}_%j.log" \
    --wrap "export PYTHONPATH=\$(pwd)/src; $PY src/dendritic_modeling/scripts/training/train_experiments.py $SHOW/${arm}_config.yaml"
done

echo "All three submitted. Watch them side by side in Kempner Insight:"
echo "  same start, same resources, same work -- triton ends ~67 min early,"
echo "  fused ~51 min early, eager runs the full ~76 min."
