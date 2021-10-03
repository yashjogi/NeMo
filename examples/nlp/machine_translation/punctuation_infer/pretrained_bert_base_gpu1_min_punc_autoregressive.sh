ds_path=/home/apeganov/NeMo/examples/nlp/token_classification/data
job_results_name="2329536_pretrained_hf_bert_base_uncased_steps300000_gpu1_bs10240_lr1e-4"
output="${ds_path}/predictions_wmt/test_min_punc/${job_results_name}/pred.txt"
mkdir -p "$(dirname "${output}")"
python nmt_transformer_infer.py \
    --model="~/NeMo/examples/nlp/machine_translation/ngc_results/aayn_base_min_punc_autoregressive/${job_results_name}/nemo_experiments/AAYNBase/best.nemo" \
    --srctext="${ds_path}/wmt/test/text_test.txt" \
    --tgtout="${output}" \
    --target_lang en \
    --source_lang en \
    --word_tokens U O \
    --add_src_num_words_to_batch