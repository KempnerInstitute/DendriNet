#!/bin/bash
# LEARNING-level overlay across the gather arms: five REAL training runs of
# the same config -- loop, loop_ctrl (the noise floor), fused, triton, cuda --
# recording loss trajectories, wall-clock, and peak GPU memory per arm.
# The twins come from gen_gather_arm_twins.py (identical except gather flags,
# rewiring frozen in every arm). loop_ctrl reruns loop.yaml under a different
# run name: the loop-vs-loop_ctrl gap DEFINES "identical learning"; every
# other arm is judged against that floor, not against zero.
#
#   bash finalized/harness/learn_overlay.sh <twindir> <outdir>
#
# Resume-safe: arms with a training_summary.json are skipped on rerun.
set -uo pipefail
TWINS=${1:?twindir}
OUT=${2:?outdir}
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."   # repo root
source finalized/setup_cuda_env.sh
export PYTHONPATH="src:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tee "$OUT/gpu.txt"

arm() { # arm <tag> <config>
  local tag="$1" cfg="$2"
  if ls "$OUT/$tag"/*/training_summary.json >/dev/null 2>&1; then
    echo "SKIP $tag"; return; fi
  echo "== $tag =="
  local pf; pf=$(mktemp)
  ( m=0; while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); case "$u" in ""|*[!0-9]*);; *) [ "$u" -gt "$m" ] && m=$u; echo "$m" > "$pf";; esac; sleep 0.5; done ) & local p=$!
  local t0=$SECONDS
  python src/dendritic_modeling/scripts/training/train_experiments.py "$cfg" \
    --output_dir "$OUT/$tag" --run_name "$tag" > "$OUT/$tag.log" 2>&1
  local rc=$? wall=$((SECONDS - t0))
  kill $p 2>/dev/null
  local path_line
  path_line=$(grep -m1 "gather path resolved" "$OUT/$tag.log" | sed 's/.*resolved: //')
  echo "$tag rc=$rc wall_s=$wall peak_mb=$(cat "$pf" 2>/dev/null) path=${path_line:-unknown}" | tee -a "$OUT/arms.txt"
  rm -f "$pf"
}

arm loop      "$TWINS/loop.yaml"
arm loop_ctrl "$TWINS/loop.yaml"
arm fused     "$TWINS/fused.yaml"
arm triton    "$TWINS/triton.yaml"
arm cuda      "$TWINS/cuda.yaml"

python - "$OUT" <<'PYEOF'
import glob, json, sys
out = sys.argv[1]
tags = ("loop", "loop_ctrl", "fused", "triton", "cuda")
traj, meta = {}, {}
for tag in tags:
    p = sorted(glob.glob(f"{out}/{tag}/*/training_summary.json"))
    if p:
        d = json.load(open(p[-1]))
        traj[tag] = {"train": d["train_losses"], "valid": d["valid_losses"]}
for line in open(f"{out}/arms.txt"):
    f = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
    meta[line.split()[0]] = f
print(f"\n===== LEARNING OVERLAY {open(out+'/gpu.txt').read().strip()} =====")
for tag in tags:
    if tag in traj:
        t = traj[tag]
        print(f"{tag:>10} train {['%.4f' % x for x in t['train']]}")
        print(f"{'':>10} valid {['%.4f' % x for x in t['valid']]}")
verdicts = {}
if "loop" in traj and "loop_ctrl" in traj:
    z = lambda a, b: max(abs(x-y) for x, y in zip(a["train"]+a["valid"], b["train"]+b["valid"]))
    noise = z(traj["loop"], traj["loop_ctrl"])
    print(f"\nnoise floor (loop vs loop_ctrl, max |d|): {noise:.2e}")
    for tag in ("fused", "triton", "cuda"):
        if tag in traj:
            d = z(traj["loop"], traj[tag])
            ok = d <= max(noise*10, 5e-3)
            verdicts[tag] = {"max_dev": d, "within": ok}
            print(f"{tag:>10} vs loop, max |d|: {d:.2e}   "
                  f"{'WITHIN ~10x noise' if ok else 'EXCEEDS'}")
    verdicts["noise_floor"] = noise
json.dump({"traj": traj, "meta": meta, "verdicts": verdicts},
          open(f"{out}/overlay.json", "w"), indent=1)
print(f"\nwrote {out}/overlay.json")
PYEOF
