ds_path=/media/apeganov/DATA/punctuation_and_capitalization/simplest/wmt_92_128_14.12.2021
output="${ds_path}/preds/nmt_wmt_large6x6_bs400000_steps300000_lr2e-4/preds.txt"
model_path="/home/apeganov/NWInf_results/autoregressive_punctuation_capitalization/nmt_wmt_large6x6_bs400000_steps300000_lr2e-4/checkpoints/AAYNLarge6x6.nemo"
mkdir -p "$(dirname "${output}")"
python nmt_transformer_infer.py \
    --model=${model_path} \
    --srctext="${ds_path}/for_upload/IWSLT_tst2019/input.txt" \
    --tgtout="${output}" \
    --word_tokens u U O \
    --add_src_num_words_to_batch \
    --cuda_device 1

cd punctuation_infer
python compute_metrics.py \
    --hyp ${output} \
    --ref "${ds_path}/for_upload/IWSLT_tst2019/autoregressive_labels.txt" \
    --output "${ds_path}/preds/nmt_wmt_large6x6_bs400000_steps300000_lr2e-4/scores.json" \
    --normalize_punctuation_in_hyp
cd -
set +e