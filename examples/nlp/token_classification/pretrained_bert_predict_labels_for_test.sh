python punctuate_capitalize_infer.py \
  --input_text ~/NeMo/examples/nlp/token_classification/data/wmt/test/text_test.txt \
  --output_text ~/NeMo/examples/nlp/token_classification/data/predictions_wiki/evelina_model_with_quotes_draco_rno139405/best/pred.txt \
  --model_path '~/NWInf_results/autoregressive_punctuation_capitalization/evelina_wiki_with_quotes_draco/checkpoints/Punctuation_and_Capitalization.nemo' \
  --max_seq_length 64 \
  --step 8 \
  --margin 16 \
  --save_labels_instead_of_text