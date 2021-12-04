python test_seq_len_step_margin.py \
  --model_path ~/NWInf_results/autoregressive_punctuation_capitalization/evelina_wiki_wmt_large/results_iwslt_tst2019_punctuation_and_capitalization_BERT_large_lr6e-5_warmupratio0.06_steps100k_wasrestore_4.12.2021.nemo \
  --labels ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/labels_iwslt_en_text.txt \
  --source_text ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/text_iwslt_en_text.txt \
  --output_dir ~/NWInf_results/autoregressive_punctuation_capitalization/evelina_wiki_wmt_large/param_selection_for_punctuation_punctuation_and_capitalization_BERT_large_lr6e-5_warmupratio0.06_steps100k_wasrestore_4.12.2021 \
  --max_seq_length 32 48 64 92 128 150 \
  --margin 8 12 16 24 \
  --step 4 8 14
