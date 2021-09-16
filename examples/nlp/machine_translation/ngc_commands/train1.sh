WANDB_API_KEY="$1"

read -r -d '' command << EOF
cd NeMo
git checkout translation_punc
export PYTHONPATH="$(pwd)"
cd examples/nlp/machine_translation
wandb login ${WANDB_API_KEY}
python enc_dec_nmt.py --config-path=conf --config-name aayn_base_punc
EOF

ngc batch run \
  --instance dgx1v.16g.8.norm \
  --name "Autoreg_punc_cap_orig_mt" \
  --image "pytorch:21.08-py3" \
  --result /result \
  --datasetid 88130:/data \
  --commandline "${command}"