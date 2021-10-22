python punctuation_capitalization_train.py \
    --config-path=conf/small_wiki \
    --config-name small_wiki_train_sorted_local \
    trainer.gpus=1 \
    exp_manager.create_wandb_logger=false
