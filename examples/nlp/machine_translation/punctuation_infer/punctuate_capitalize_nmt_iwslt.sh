python punctuate_capitalize_nmt.py \
    --input_text ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019/for_testing_punctuation_model/one_line_text_iwslt_en_text.txt \
    --output_text debug_punctuate_capitalize_nmt.txt \
    --model_path ~/NWInf_results/autoregressive_punctuation_capitalization/nmt_wiki_wmt_large6x6_bs400000_steps400000_lr2e-4/checkpoints/AAYNLarge6x6.nemo \
    --max_seq_length 32 \
    --step 28 \
    --margin 2 \
    --batch_size 128 \
    --no_all_upper_label \
    --add_source_num_words_to_batch