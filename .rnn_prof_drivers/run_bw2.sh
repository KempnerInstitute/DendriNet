set -x
cd /n/home12/rwhalen/DendriNet
export PYTHONPATH=/n/home12/rwhalen/DendriNet/src
PY=/n/home12/rwhalen/.conda/envs/dendritic_env/bin/python
CFG=profilings/dendrinet_harness/scale_ab/scale_5k_5k_off.yaml
OUT=profilings/dendrinet_harness/rnn_prof/blackwell
echo "########## index_add_ microbench ##########"
$PY .rnn_prof_drivers/bench_index_add.py
for P in loop fused; do
  echo "########## BLACKWELL cudagraph ceiling path=$P ##########"
  $PY profile_rnn_gather.py --config $CFG --path $P --mode cudagraph \
     --batch 1 --seq-len 58 --iters 10 --outdir $OUT
done
echo "########## BW2 DONE ##########"
