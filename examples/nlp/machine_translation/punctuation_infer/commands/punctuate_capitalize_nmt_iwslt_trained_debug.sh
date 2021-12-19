set -e
ds_path=/media/apeganov/DATA/punctuation_and_capitalization/simplest/wmt_92_128_14.12.2021
output_text="${ds_path}/preds/debug_inference/pred_text.txt"
output_labels="${ds_path}/preds/debug_inference/pred_labels.txt"
model_path="/home/apeganov/NWInf_results/autoregressive_punctuation_capitalization/nmt_wmt_large6x6_bs400000_steps300000_lr2e-4/checkpoints/AAYNLarge6x6.nemo"
python punctuate_capitalize_nmt.py \
    --input_text "debug/short_input.txt" \
    --output_text "${output_text}" \
    --output_labels "${output_labels}" \
    --model_path "${model_path}" \
    --max_seq_length 20 \
    --step 10 \
    --margin 1 \
    --batch_size 256 \
    --no_all_upper_label \
    --add_source_num_words_to_batch \
    --make_queries_contain_intact_sentences

python compute_metrics.py \
    --hyp ${output_labels} \
    --ref "${ds_path}/for_upload/IWSLT_tst2019/autoregressive_labels.txt" \
    --output "${ds_path}/preds/debug_inference/scores.json" \
    --normalize_punctuation_in_hyp
set +e