python prepare_big_data_for_punctuation_capitalization_task.py \
  --output_dir /media/apeganov/DATA/prepared_wiki_unsplit \
  --corpus_types wikipedia \
  --clean_data_dir /media/apeganov/DATA/debug_punct_wiki_preparation_clean \
  --create_model_input \
  --bert_labels \
  --autoregressive_labels \
  --allowed_punctuation '.,?' \
  --fasttext_model lid.176.bin \
  --only_first_punctuation_character_after_word_in_autoregressive \
  --no_label_if_all_characters_are_upper_case \
  ~/data/enwiki-20210920-pages-articles-multistream.xml