ds_path=/media/apeganov/DATA/punctuation_and_capitalization/simplest/wiki_wmt_92_128_29.11.2021
output="${ds_path}/preds_nmt_wiki_wmt_large6x6_bs400000_steps400000_lr2e-4_cross/pred_cross_labels.txt"
model_path="ngc_results/aayn_base_min_punc_cross/2371822_from_scratch_original_steps300000_gpu1/nemo_experiments/AAYNBase/2021-10-27_18-00-01/checkpoints/AAYNBase.nemo"
mkdir -p "$(dirname "${output}")"
python nmt_transformer_infer.py \
    --model="${model_path}" \
    --srctext="${ds_path}/for_upload/IWSLT_tst2019/input.txt" \
    --tgtout="${output}" \
    --target_lang en \
    --source_lang en \
    --word_tokens a b c d e f g h i j k l m n o p q \
    --add_src_num_words_to_batch
