WANDB_API_KEY="$1"

read -r -d '' command << EOF
set -e -x
OMP_NUM_THREADS=8
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
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased'); tokenizer.save_pretrained('pretrained_tokenizer')"
python punctuation_capitalization_train.py --config-path=conf \
    --config-name wmt_train_from_scratch \
    trainer.gpus=1
set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.1.norm \
  --name "ml-model.bert punctuation_capitalization_training_on_wmt" \
  --image "nvidia/pytorch:21.08-py3" \
  --result /result \
  --datasetid 88512:/data \
  --commandline "${command}"