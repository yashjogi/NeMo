python test_seq_len_step_margin.py \
  --model_path ~/NWInf_results/autoregressive_punctuation_capitalization/evelina_wiki_with_quotes_draco/checkpoints/Punctuation_and_Capitalization.nemo \
  --labels ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/labels_iwslt_en_text.txt \
  --source_text ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/text_iwslt_en_text.txt \
  --output_dir ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/evelina_model_with_quotes_draco_rno139405 \
  --max_seq_length 32 48 64 92 128 \
  --margin 8 12 16 \
  --step 4 8 14
