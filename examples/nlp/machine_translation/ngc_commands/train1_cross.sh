WANDB_API_KEY="$1"

read -r -d '' command << EOF
set -e -x
mkdir /result/nemo_experiments
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout feat/punc_tarred
bash reinstall.sh
cd examples/nlp/machine_translation
wandb login ${WANDB_API_KEY}
python create_autoregressive_char_vocabulary.py \
  --input /data/train/cross_labels.txt \
  --output /workspace/cross_labels_char_vocab.txt \
  --characters_to_exclude $'\n' \
  --eos_token EOS \
  --pad_token PAD
python enc_dec_nmt.py \
  --config-path=conf \
  --config-name aayn_base_cross_labels_punc_cap \
  trainer.gpus=1
set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.1.norm \
  --name "ml-model.aayn cross_labels_training" \
  --image "nvidia/pytorch:21.08-py3" \
  --result /result \
  --datasetid 90228:/data \
  --commandline "${command}"