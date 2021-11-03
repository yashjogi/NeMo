python create_punctuation_capitalization_tarred_dataset.py \
  --text /media/apeganov/DATA/prepared_wiki_wmt_unsplit_48_65_3.11.2021/train/input.txt \
  --labels /media/apeganov/DATA/prepared_wiki_wmt_unsplit_48_65_3.11.2021/train/bert_labels.txt \
  --output_dir /media/apeganov/DATA/prepared_wiki_wmt_unsplit_48_65_3.11.2021/train_bert_tarred \
  --lines_per_dataset_fragment 500000 \
  --num_batches_per_tarfile 50
