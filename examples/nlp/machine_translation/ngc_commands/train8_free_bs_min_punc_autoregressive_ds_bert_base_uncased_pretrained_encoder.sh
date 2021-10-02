WANDB_API_KEY="$1"

read -r -d '' command << EOF
set -e -x
OMP_NUM_THREADS=8
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
python enc_dec_nmt.py --config-path=conf \
  --config-name aayn_base_min_punc_autoregressive_ds_bert_base_uncased_pretrained_encoder_ngc \
  trainer.gpus=8 \
  model.train_ds.tokens_in_batch=9196 \
  model.validation_ds.tokens_in_batch=9196 \
  model.test_ds.tokens_in_batch=9196
set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.8.norm \
  --name "ml-model.aayn min_autoreg_punc_cap_mt_bert_base_uncased_pretrained_encoder" \
  --image "nvidia/pytorch:21.08-py3" \
  --result /result \
  --datasetid 88728:/data \
  --commandline "${command}"