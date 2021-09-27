python test_seq_len_step_margin.py \
  --model_path '~/NeMo/examples/nlp/token_classification/ngc_results/wmt/min_punc/2312172_lr1e-4_steps300000_gpu1/final.nemo' \
  --labels ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/labels_iwslt_en_text.txt \
  --source_text ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/text_iwslt_en_text.txt \
  --output_dir scores_wmt_posttrained