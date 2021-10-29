python punctuate_capitalize_infer.py \
  --input_text ~/NeMo/examples/nlp/token_classification/data/wmt/test/text_test.txt \
  --output_text ~/NeMo/examples/nlp/token_classification/data/predictions_wmt/train_on_wiki/evelina_model_with_quotes_draco_rno139405/best/pred.txt \
  --model_path '~/NWInf_results/autoregressive_punctuation_capitalization/evelina_wiki_with_quotes_draco/checkpoints/Punctuation_and_Capitalization.nemo' \
  --max_seq_length 512 \
  --step 510 \
  --margin 0 \
  --save_labels_instead_of_text

python punctuate_capitalize_infer.py \
  --input_text ~/NeMo/examples/speech_translation/punct_prepared_datasets/prepared_punctuation_data_min_punctuation_27.10.2021_3.13/test/input.txt \
  --output_text ~/NeMo/examples/speech_translation/punct_prepared_datasets/prepared_punctuation_data_min_punctuation_27.10.2021_3.13/test_predictions/evelina_wiki_with_quotes_draco_rno139405/best/pred_text.txt \
  --model_path '~/NWInf_results/autoregressive_punctuation_capitalization/evelina_wiki_with_quotes_draco/checkpoints/Punctuation_and_Capitalization.nemo' \
  --max_seq_length 512 \
  --step 510 \
  --margin 0