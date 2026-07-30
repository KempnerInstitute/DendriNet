set -x
cd /n/home12/rwhalen/DendriNet
export PYTHONPATH=/n/home12/rwhalen/DendriNet/src
PY=/n/home12/rwhalen/.conda/envs/dendritic_env/bin/python
CFG=profilings/dendrinet_harness/scale_ab/scale_5k_5k_off.yaml
OUT=profilings/dendrinet_harness/rnn_prof/a100
mkdir -p $OUT
for P in loop fused; do
  echo "########## A100 path=$P profile T=58 ##########"
  $PY profile_rnn_gather.py --config $CFG --path $P --mode profile \
     --batch 1 --seq-len 58 --warmup 2 --iters 3 --prof-iters 2 --outdir $OUT
done
for P in loop fused; do
  echo "########## A100 path=$P batch sweep ##########"
  $PY profile_rnn_gather.py --config $CFG --path $P --mode sweep \
     --batches 1 2 4 8 16 --seq-len 58 --warmup 1 --iters 2 --outdir $OUT
done
echo "########## A100 DONE ##########"
