python prepare_big_data_for_punctuation_capitalization_task_simple.py \
  --output_dir /media/apeganov/DATA/debug_europarl_preparation \
  --corpus_types europarl \
  --create_model_input \
  --bert_labels \
  --autoregressive_labels \
  --allowed_punctuation '.,?' \
  --only_first_punctuation_character_after_word_in_autoregressive \
  --no_label_if_all_characters_are_upper_case \
  --input_files ~/data/europarl/v10/training-monolingual/europarl-v10.en.tsv \
  --num_jobs 24 #\
  #--resume_from writing