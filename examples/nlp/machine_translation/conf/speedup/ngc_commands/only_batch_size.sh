WANDB_API_KEY="$1"

read -r -d '' command << EOF
set -e -x
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
python enc_dec_nmt.py --config-path=conf/speedup \
  --config-name original \
  trainer.gpus=1 \
  model.train_ds.tokens_in_batch=38000 \
  model.validation_ds.tokens_in_batch=512 \
  model.test_ds.tokens_in_batch=512 \
  exp_manager.wandb_logger_kwargs.name=batch_size_only
set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.1.norm \
  --name "ml-model.aayn speedup_batch_size_only" \
  --image "nvidia/pytorch:21.08-py3" \
  --result /result \
  --datasetid 88728:/data \
  --commandline "${command}"