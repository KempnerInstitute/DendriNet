set -x
cd /n/home12/rwhalen/DendriNet
export PYTHONPATH=/n/home12/rwhalen/DendriNet/src
PY=/n/home12/rwhalen/.conda/envs/dendritic_env/bin/python
CFG=profilings/dendrinet_harness/scale_ab/scale_5k_5k_off.yaml
OUT=profilings/dendrinet_harness/rnn_prof/blackwell_fp32acc
mkdir -p $OUT
echo "########## BW equivalence check ##########"
$PY .rnn_prof_drivers/check_fused_equiv.py
echo "########## BW index_add_ microbench (re-confirm) ##########"
$PY .rnn_prof_drivers/bench_index_add.py
echo "########## BW fused AFTER fp32-acc fix ##########"
$PY profile_rnn_gather.py --config $CFG --path fused --mode profile \
   --batch 1 --seq-len 58 --warmup 2 --iters 5 --prof-iters 2 --outdir $OUT
echo "########## BW fused sweep AFTER fix ##########"
$PY profile_rnn_gather.py --config $CFG --path fused --mode sweep \
   --batches 1 2 4 8 16 32 --seq-len 58 --warmup 1 --iters 3 --outdir $OUT
echo "########## BW3 DONE ##########"
