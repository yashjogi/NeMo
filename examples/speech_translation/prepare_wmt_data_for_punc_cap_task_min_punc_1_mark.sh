python prepare_wmt_data_for_punctuation_capitalization_task.py \
  europarl/v10/training-monolingual/europarl-v10.en.tsv \
  news-commentary/v16/training-monolingual/news-commentary-v16.en \
  rapid/RAPID_2019.de-en.xlf \
  TED_Talks/en-ja/train.tags.en-ja.en \
  --input_language en \
  --output_dir prepared_punctuation_data_min_punctuation_26.9.2021_19.00 \
  --corpus_types europarl 'news-commentary' rapid TED \
  --clean_data_dir cleaned_wmt_min_punc \
  --create_model_input \
  --autoregressive_labels \
  --bert_labels \
  --allowed_punctuation ',.?' \
  --only_first_punctuation_character_after_word_in_autoregressive
