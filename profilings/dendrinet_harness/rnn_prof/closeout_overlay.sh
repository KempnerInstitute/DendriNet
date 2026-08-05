#!/bin/bash
# Closing trajectory overlay for the feedforward single-GPU engineering track.
#
# Four arms, one trainer, one seed set, 5 real epochs each:
#   eager         reactivation stack OFF, cuda gather   (the pre-optimization math)
#   eager_control reactivation stack OFF, cuda gather   (run twice: the noise floor)
#   react_fused   reactivation stack ON,  cuda gather   (the 2.19x memory stack)
#   gather_fused  reactivation stack ON,  FUSED gather  (different gather KERNEL)
#
# Claim under test: none of the optimizations changes what training does. The
# control arm defines "same" -- any fused-vs-eager gap must be judged against the
# eager-vs-eager gap, not against zero. Run on BOTH architectures.
#   bash closeout_overlay.sh <outdir>
set -uo pipefail
OUT=${1:?outdir}
mkdir -p "$OUT"
export PYTHONPATH="src:${PYTHONPATH:-}"
PY=python
CFG_CUDA=configs/examples/kernel_stress_ab/kernel_stress_3L_16k_cuda_learn.yaml
CFG_FUSED=configs/examples/kernel_stress_ab/kernel_stress_3L_16k_fusedgather_learn.yaml
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tee "$OUT/gpu.txt"

run() { # run <tag> <sigmoid-flag> <config>
  local tag="$1"; local flag="$2"; local cfg="$3"
  if [ -s "$OUT/$tag.log" ] && grep -q "Best epoch" "$OUT/$tag.log"; then
    echo "SKIP $tag"; return; fi
  echo "== $tag =="
  DENDRINET_POSITIVE_TANH_SIGMOID=$flag $PY -u \
    src/dendritic_modeling/scripts/training/train_experiments.py "$cfg" \
    --output_dir "$OUT/$tag" --run_name "$tag" > "$OUT/$tag.log" 2>&1
  grep -E "gather path resolved|epoch=5/5" "$OUT/$tag.log" | tail -2
}
run eager 0 "$CFG_CUDA"
run eager_control 0 "$CFG_CUDA"
run react_fused 1 "$CFG_CUDA"
run gather_fused 1 "$CFG_FUSED"

$PY - "$OUT" <<'PYEOF'
import glob, json, sys
out = sys.argv[1]
print(f"\n===== OVERLAY {open(out+'/gpu.txt').read().strip()} =====")
traj = {}
for tag in ("eager", "eager_control", "react_fused", "gather_fused"):
    p = sorted(glob.glob(f"{out}/{tag}/{tag}_*/training_summary.json"))
    if p:
        d = json.load(open(p[-1]))
        traj[tag] = (d["train_losses"], d["valid_losses"])
        print(f"{tag:>14} train {['%.4f' % x for x in d['train_losses']]}")
        print(f"{'':>14} valid {['%.4f' % x for x in d['valid_losses']]}")
if "eager" in traj and "eager_control" in traj:
    noise = max(abs(a-b) for a,b in zip(traj["eager"][0]+traj["eager"][1],
                                        traj["eager_control"][0]+traj["eager_control"][1]))
    print(f"\nnoise floor (eager vs control, max |d|): {noise:.2e}")
    for tag in ("react_fused", "gather_fused"):
        if tag in traj:
            d = max(abs(a-b) for a,b in zip(traj["eager"][0]+traj["eager"][1],
                                            traj[tag][0]+traj[tag][1]))
            print(f"{tag:>14} vs eager, max |d|: {d:.2e}   "
                  f"{'WITHIN ~10x noise' if d <= max(noise*10, 5e-3) else 'EXCEEDS'}")
PYEOF
