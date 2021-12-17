python test_seq_len_step_margin.py \
  --model_path ~/NWInf_results/autoregressive_punctuation_capitalization/evelina_wmt_base_lr1e-4_steps150k_from_scratch_bs160k/checkpoints/Punctuation_and_Capitalization.nemo \
  --labels ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/labels_iwslt_en_text.txt \
  --source_text ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/text_iwslt_en_text.txt \
  --output_dir ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/evelina_model_base_from_scratch_wmt \
  --max_seq_length 32 48 64 92 128 \
  --margin 8 12 16 \
  --step 4 8 14 \
  --cuda_device 0