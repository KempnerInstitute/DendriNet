set -x
cd /n/home12/rwhalen/DendriNet
export PYTHONPATH=/n/home12/rwhalen/DendriNet/src
PY=/n/home12/rwhalen/.conda/envs/dendritic_env/bin/python
CFG=profilings/dendrinet_harness/scale_ab/scale_5k_5k_off.yaml
OUT=profilings/dendrinet_harness/rnn_prof/a100_fp32acc
mkdir -p $OUT
echo "########## EQUIVALENCE CHECK ##########"
$PY .rnn_prof_drivers/check_fused_equiv.py
echo "########## A100 fused AFTER fp32-acc fix ##########"
$PY profile_rnn_gather.py --config $CFG --path fused --mode profile \
   --batch 1 --seq-len 58 --warmup 2 --iters 3 --prof-iters 2 --outdir $OUT
echo "########## A100_3 DONE ##########"
