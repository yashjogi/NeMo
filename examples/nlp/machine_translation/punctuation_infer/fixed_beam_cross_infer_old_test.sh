ds_path=/home/apeganov/NeMo/examples/nlp/token_classification/data/wmt/test
output="/home/apeganov/NeMo/examples/nlp/token_classification/data/predictions_wmt/2371822_cross_labels_seq2seq_from_scratch_original_steps300000_gpu1/pred_cross_labels.txt"
model_path="ngc_results/aayn_base_min_punc_cross/2371822_from_scratch_original_steps300000_gpu1/nemo_experiments/AAYNBase/2021-10-27_18-00-01/checkpoints/AAYNBase.nemo"
mkdir -p "$(dirname "${output}")"
python nmt_transformer_infer.py \
    --model="${model_path}" \
    --srctext="${ds_path}/test/input.txt" \
    --tgtout="${output}" \
    --target_lang en \
    --source_lang en \
    --word_tokens a b c d e f g h i j k l m n o p q \
    --add_src_num_words_to_batch
