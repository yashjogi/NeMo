for asr in stt_en_citrinet_1024 stt_en_citrinet_1024_gamma_0_25 "/home/apeganov/checkpoints/CitriNet-1024-8x-Stride-Gamma-0.25.nemo" "/home/apeganov/checkpoints/Citrinet-1024-SPE-Unigram-1024-Jarvis-ASRSet-2.0_no_weight_decay_e250-averaged.nemo"; do
  bash test_iwslt.sh ~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019-5 \
    "${asr}" \
    '~/NeMo/examples/nlp/token_classification/ngc_results/wmt/min_punc/2312172_lr1e-4_steps300000_gpu1/final.nemo' \
    /home/apeganov/checkpoints/wmt21_en_de_backtranslated_24x6_averaged.nemo \
    /home/apeganov/iwslt_2019_test_result_wmt_posttrained_punctuation \
    0 \
    1
 done
