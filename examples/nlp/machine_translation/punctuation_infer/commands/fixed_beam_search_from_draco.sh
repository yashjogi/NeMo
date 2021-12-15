ds_path=/media/apeganov/DATA/punctuation_and_capitalization/simplest/wiki_wmt_92_128_29.11.2021
output="${ds_path}/preds/nmt_wiki_wmt_large6x6_bs400000_steps400000_lr2e-4/preds.txt"
model_path="/home/apeganov/NWInf_results/autoregressive_punctuation_capitalization/nmt_wiki_wmt_large6x6_bs400000_steps400000_lr2e-4/checkpoints/AAYNLarge6x6.nemo"
mkdir -p "$(dirname "${output}")"
python nmt_transformer_infer.py \
    --model=${model_path} \
    --srctext="${ds_path}/for_upload/IWSLT_tst2019/input.txt" \
    --tgtout="${output}" \
    --word_tokens u U O \
    --add_src_num_words_to_batch \
    --cuda_device 1
