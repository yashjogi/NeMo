ds_path=/home/lab/NeMo/examples/speech_translation/prepared_punctuation_data_all_punctuation
output="${ds_path}/predictions/test_min_punc/free_beam_search_gpu1_steps300k_lr1e-4_bs16384_2320685/pred.txt"
mkdir -p "$(dirname "${output}")"
python nmt_transformer_infer.py \
    --model=/home/lab/NeMo/examples/nlp/machine_translation/ngc_results/aayn_base_min_punc_autoregressivefree_beam_search_gpu1_steps300k_lr1e-4_bs16384_2320685/nemo_experiments/AAYNBase/2021-09-26_22-57-10/checkpoints/AAYNBase.nemo \
    --srctext="${ds_path}/test_min_punc/input.txt" \
    --tgtout="${output}" \
    --target_lang en \
    --source_lang en \
    --word_tokens U O \
    --add_src_num_words_to_batch
