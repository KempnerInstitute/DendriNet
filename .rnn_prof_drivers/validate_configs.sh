set -x
cd /n/home12/rwhalen/DendriNet
export PYTHONPATH=/n/home12/rwhalen/DendriNet/src
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + RANDOM % 20000))
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
export NCCL_DEBUG=WARN
PY=/n/home12/rwhalen/.conda/envs/dendritic_env/bin/python
echo "########## config load check ##########"
$PY -c "
from dendritic_modeling.config import load_config
for p in ['configs/dendritic_rnn/large_scale_500k_5k/base.yaml',
          'configs/dendritic_rnn/large_scale_500k_5k/canary.yaml']:
    load_config(p); print('loaded OK:', p)
"
echo "########## canary training run (fused ON) ##########"
$PY src/dendritic_modeling/scripts/training/train_experiments_fsdp.py \
   configs/dendritic_rnn/large_scale_500k_5k/canary.yaml 2>&1 | grep -E "Train Loss|Error|Traceback|CUDA" | tail -5
echo "########## VALIDATE DONE ##########"
