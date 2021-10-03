python punctuate_capitalize_infer.py \
  --input_text ~/NeMo/examples/nlp/token_classification/data/wmt/test/text_test.txt \
  --output_text ~/NeMo/examples/nlp/token_classification/data/predictions_wmt/2329633_evelina_model_from_scratch/pred.txt \
  --model_path '~/NeMo/examples/nlp/token_classification/ngc_results/wmt/min_punc/2329633_lr1e-4_steps300000_gpu1_from_scratch/final.nemo' \
  --max_seq_length 512 \
  --step 510 \
  --margin 0 \
  --save_only_labels