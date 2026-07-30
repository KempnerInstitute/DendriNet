set -x
cd /n/home12/rwhalen/DendriNet
export PYTHONPATH=/n/home12/rwhalen/DendriNet/src
PY=/n/home12/rwhalen/.conda/envs/dendritic_env/bin/python
CFG=profilings/dendrinet_harness/scale_ab/scale_5k_5k_off.yaml
OUT=profilings/dendrinet_harness/rnn_prof/blackwell_cuda_triton
mkdir -p $OUT
for P in cuda triton; do
  echo "########## BLACKWELL path=$P profile T=58 batch=1 ##########"
  timeout 3000 $PY profile_rnn_gather.py --config $CFG --path $P --mode profile \
     --batch 1 --seq-len 58 --warmup 1 --iters 2 --prof-iters 1 --outdir $OUT
  echo "exit=$?"
done
echo "########## CUDA_TRITON DONE ##########"
