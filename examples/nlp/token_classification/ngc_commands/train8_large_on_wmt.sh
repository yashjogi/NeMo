WANDB_API_KEY="$1"

read -r -d '' command << EOF
set -e -x
OMP_NUM_THREADS=1
git clone https://github.com/NVIDIA/NeMo
mkdir -p /result/nemo_experiments
cd NeMo
git checkout feat/punc_tarred
source reinstall.sh
cd examples/nlp/token_classification
wandb login ${WANDB_API_KEY}
python -c "from nemo.collections.nlp.modules import get_tokenizer;get_tokenizer('bert-large-uncased', use_fast=False)"
python punctuation_capitalization_train.py --config-path=conf \
    --config-name wmt_train \
    trainer.gpus=8 \
    model.language_model.pretrained_model_name=bert-large-uncased \
    model.train_ds.tokens_in_batch=8000 \
    model.validation_ds.batch_size=8000 \
    model.test_ds.batch_size=8000
set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.8.norm \
  --name "ml-model.bert large_punctuation_capitalization_training_on_wmt" \
  --image "nvidia/pytorch:21.08-py3" \
  --result /result \
  --datasetid None:/data \
  --commandline "${command}"