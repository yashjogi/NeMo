WANDB_API_KEY="$1"

read -r -d '' command << EOF
set -e -x
OMP_NUM_THREADS=8
cd NeMo
git checkout feat/punc_tarred
git pull
bash reinstall.sh
mkdir -p /result/nemo_experiments
cd examples/nlp/token_classification
wandb login ${WANDB_API_KEY}
python punctuation_capitalization_train.py --config-path=conf \
    --config-name wmt_train \
    trainer.gpus=1
set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.1.norm \
  --name "ml-model.bert new_punctuation_capitalization_training_on_wmt" \
  --image "nvcr.io/nvidian/ac-aiapps/speech_translation:latest" \
  --result /result \
  --datasetid 90512:/data \
  --commandline "${command}"