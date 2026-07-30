set -x
cd /n/home12/rwhalen/DendriNet
export PYTHONPATH=/n/home12/rwhalen/DendriNet/src
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + RANDOM % 20000))
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
export NCCL_DEBUG=WARN
PY=/n/home12/rwhalen/.conda/envs/dendritic_env/bin/python
for CFG in scale_5k_5k_off scale_5k_5k_fused; do
  echo "########## EPOCH A/B $CFG ##########"
  $PY src/dendritic_modeling/scripts/training/train_experiments_fsdp.py \
     profilings/dendrinet_harness/scale_ab/$CFG.yaml 2>&1 | grep -E "Train Loss|Time:|Error|Traceback" | tail -5
done
echo "########## EPOCH_AB DONE ##########"
