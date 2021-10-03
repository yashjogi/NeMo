python punctuate_capitalize_infer.py \
  --input_text ~/NeMo/examples/nlp/token_classification/data/wmt/test/text_test.txt \
  --output_text ~/NeMo/examples/nlp/token_classification/data/predictions_wmt/evelina_model_job2312172/pred.txt \
  --model_path '~/NeMo/examples/nlp/token_classification/ngc_results/wmt/min_punc/2312172_lr1e-4_steps300000_gpu1/final.nemo' \
  --max_seq_length 512 \
  --step 510 \
  --margin 0 \
  --save_only_labels