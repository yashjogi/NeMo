WANDB_API_KEY="$1"

read -r -d '' command << EOF
set -e -x
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout translation_punc
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_lightning.txt
pip install -r requirements/requirements_test.txt
pip install -r requirements/requirements_nlp.txt
export PYTHONPATH="\$(pwd)"
cd examples/nlp/machine_translation
wandb login ${WANDB_API_KEY}
python enc_dec_nmt.py --config-path=conf --config-name aayn_base_punc
set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.1.norm \
  --name "ml-model.aayn autoreg_punc_cap_orig_mt" \
  --image "nvidia/pytorch:21.08-py3" \
  --result /result \
  --datasetid 88130:/data \
  --commandline "${command}"