WANDB_API_KEY="$1"

read -r -d '' command << EOF
set -e -x
export OMP_NUM_THREADS=1
mkdir /result/nemo_experiments
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout iwslt_cascade
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_lightning.txt
pip install -r requirements/requirements_test.txt
pip install -r requirements/requirements_nlp.txt
export PYTHONPATH="\$(pwd)"
cd examples/nlp/machine_translation
wandb login ${WANDB_API_KEY}
python enc_dec_nmt.py \
  --config-path=conf/speedup \
  --config-name original \
  trainer.gpus=1 \
  exp_manager.wandb_logger_kwargs.name=omp_num_threads_1_only

set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.1.norm \
  --name "ml-model.aayn speedup_omp_num_threads_1_only" \
  --image "nvidia/pytorch:21.08-py3" \
  --result /result \
  --datasetid 88728:/data \
  --commandline "${command}"