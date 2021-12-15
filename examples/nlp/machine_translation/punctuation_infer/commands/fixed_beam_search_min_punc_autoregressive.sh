ds_path=/home/apeganov/NeMo/examples/nlp/token_classification/data
output="${ds_path}/predictions_wmt/test_min_punc/free_beam_search_gpu1_steps300k_lr1e-4_bs16384_2320685/pred.txt"
mkdir -p "$(dirname "${output}")"
python nmt_transformer_infer.py \
    --model=/home/apeganov/NeMo/examples/nlp/machine_translation/ngc_results/aayn_base_min_punc_autoregressive/free_beam_search_gpu1_steps300k_lr1e-4_bs16384_2320685/nemo_experiments/AAYNBase/2021-09-26_22-57-10/checkpoints/AAYNBase.nemo \
    --srctext="${ds_path}/wmt/test/text_test.txt" \
    --tgtout="${output}" \
    --target_lang en \
    --source_lang en \
    --word_tokens U O \
    --add_src_num_words_to_batch
