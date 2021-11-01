WANDB_API_KEY="$1"

read -r -d '' command << EOF
set -e -x
mkdir /result/nemo_experiments
cd NeMo
git pull
git checkout decoder_with_tips
bash reinstall.sh
pip install pytorch-lightning==1.5.0rc1
cd examples/nlp/machine_translation
wandb login ${WANDB_API_KEY}
python enc_dec_nmt.py \
  --config-path=conf \
  --config-name tips_aayn_base_min_punc_autoregressive_ngc \
  trainer.gpus=1
set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.1.norm \
  --name "ml-model.aayn tips_simple" \
  --image "nvcr.io/nvidian/ac-aiapps/speech_translation:latest" \
  --result /result \
  --datasetid 90546:/data \
  --commandline "${command}"