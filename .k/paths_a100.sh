cd /n/home12/rwhalen/DendriNet
source profilings/weight_transform_opt/setup_cuda_env.sh >/dev/null 2>&1
export PYTHONPATH=/n/home12/rwhalen/DendriNet/src
/n/home12/rwhalen/.conda/envs/dendritic_env/bin/python \
  profilings/dendrinet_harness/rnn_prof/bench_gather_paths.py
echo "########## PATHS DONE ##########"
