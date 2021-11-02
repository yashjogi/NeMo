python prepare_big_data_for_punctuation_capitalization_task_simple.py \
  --output_dir /media/apeganov/DATA/debug_news_commentary_preparation \
  --corpus_types news-commentary \
  --create_model_input \
  --bert_labels \
  --autoregressive_labels \
  --allowed_punctuation '.,?' \
  --only_first_punctuation_character_after_word_in_autoregressive \
  --no_label_if_all_characters_are_upper_case \
  --input_files ~/data/news-commentary/v16/training-monolingual/news-commentary-v16.en \
  --num_jobs 24 #\
  #--resume_from writing