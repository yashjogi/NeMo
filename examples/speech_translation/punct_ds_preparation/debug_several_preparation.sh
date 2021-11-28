python prepare_big_data_for_punctuation_capitalization_task_simple.py \
  --output_dir /media/apeganov/DATA/debug_several_preparation \
  --corpus_types TED \
  --create_model_input \
  --bert_labels \
  --autoregressive_labels \
  --sequence_length_range 48 65 \
  --allowed_punctuation '.,?' \
  --only_first_punctuation_character_after_word_in_autoregressive \
  --no_label_if_all_characters_are_upper_case \
  --input_files ~/data/TED_Talks/en-ja/train.tags.en-ja.en \
  --num_jobs 24

#  --corpus_types europarl news-commentary \
#  --input_files ~/data/europarl/v10/training-monolingual/europarl-v10.en.tsv \
#      ~/data/news-commentary/v16/training-monolingual/news-commentary-v16.en \