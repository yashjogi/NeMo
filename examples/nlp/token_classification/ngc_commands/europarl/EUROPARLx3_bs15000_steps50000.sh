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
python punctuation_capitalization_train_evaluate.py --config-path=conf/ted \
    --config-name local_bs15000_steps50000 \
    exp_manager.exp_dir=/result \
    exp_manager.wandb_logger_kwargs.name=EUROPARLx3_bs15000_steps300000 \
    model.train_ds.ds_item=/data/train \
    model.validation_ds.ds_item=[/data/IWSLT_tst2019,/data/europarl_dev,/data/wiki_dev,/data/rapid_dev,/data/news_commentary_dev] \
    model.test_ds.ds_item=/data/IWSLT_tst2019 \
    trainer.gpus=1 \
    trainer.max_steps=300000 \
    +trainer.val_check_interval=5000 \
    model.optim.lr=6e-5 \
    model.optim.sched.warmup_ratio=0.04 \
    +model.train_ds.label_info_save_dir=/result/train_label_ids \
    +model.validation_ds.label_info_save_dir=/result/validation_label_ids \
    +model.test_ds.label_info_save_dir=/result/test_label_ids
set +e +x
EOF

ngc batch run \
  --instance dgx1v.16g.1.norm \
  --name "ml-model.bert EUROPARLx3_punctuation_capitalization_bs15000_steps300000" \
  --image "nvcr.io/nvidian/ac-aiapps/speech_translation:latest" \
  --result /result \
  --datasetid 92270:/data \
  --commandline "${command}"