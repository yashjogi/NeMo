python punctuate_capitalize_infer.py \
  --input_manifest ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/manifest.json \
  --output_text iwslt_infer_result.txt \
  --model_path '~/NeMo/examples/nlp/token_classification/ngc_results/wmt/min_punc/2329633_lr1e-4_steps300000_gpu1_from_scratch/nemo_experiments/final.nemo' \
  --max_seq_length 92 \
  --step 8 \
  --margin 16 \
  --save_labels_instead_of_text