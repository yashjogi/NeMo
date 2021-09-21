WANDB_API_KEY="$1"

read -r -d '' command << EOF
set -e -x
git clone https://github.com/NVIDIA/NeMo
mkdir -p /result/nemo_experiments
cd NeMo
git checkout iwslt_cascade
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_lightning.txt
pip install -r requirements/requirements_test.txt
pip install -r requirements/requirements_nlp.txt
export PYTHONPATH="\$(pwd)"
cd examples/nlp/token_classification
wandb login ${WANDB_API_KEY}
python punctuation_capitalization_train.py --config-path=conf \
    --config-name wmt_train \
    trainer.gpus=1
python punctuation_capitalization_evaluate.py --config-path=conf \
    --config-name wmt_train \
    trainer.gpus=1
set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.1.norm \
  --name "ml-model.bert punctuation_capitalization_training_on_wmt" \
  --image "nvidia/pytorch:21.08-py3" \
  --result /result \
  --datasetid 88471:/data \
  --commandline "${command}"