python text_to_punc_cap_dataset.py \
  -i /home/apeganov/NeMo/examples/speech_translation/prepared_punctuation_data_all_punctuation/test/text.txt \
  -o /home/apeganov/NeMo/examples/speech_translation/prepared_punctuation_data_all_punctuation/test_min_punc \
  --only_first_punctuation_character_after_word_in_autoregressive \
  --no_label_if_all_characters_are_upper_case \
  --create_model_input \
  --bert_labels \
  --autoregressive_labels \
  --allowed_punctuation ',.?'