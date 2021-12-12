python text_to_punc_cap_dataset.py \
  --input_text /media/apeganov/DATA/punctuation_and_capitalization/simplest/wiki_wmt_92_128_29.11.2021/for_upload/IWSLT_tst2019/text.txt \
  --output_dir /media/apeganov/DATA/punctuation_and_capitalization/all/wiki_wmt_92_128_12.12.2021/for_upload/IWSLT_tst2019 \
  --create_model_input \
  --bert_labels \
  --autoregressive_labels \
  --allowed_punctuation '.,?"-;:!()'