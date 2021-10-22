python test_seq_len_step_margin.py \
  --model_path '~/NeMo/examples/nlp/token_classification/ngc_results/small_wiki/min_punc/2358708_lr1e-4_steps400000_gpu1_unsorted/nemo_experiments/Punctuation_and_Capitalization/2021-10-20_17-50-30/checkpoints/Punctuation_and_Capitalization.nemo' \
  --labels ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/labels_iwslt_en_text.txt \
  --source_text ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/text_iwslt_en_text.txt \
  --output_dir scores_small_wiki_posttrained_job2358708 \
  --max_seq_length 32 48 64 92 128 \
  --margin 8 12 16 \
  --step 4 8 14
