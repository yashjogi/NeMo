ds_path=/home/lab/NeMo/examples/speech_translation/prepared_punctuation_data_all_punctuation
output="${ds_path}/predictions/aayn_base_free_beam_search_job2308718/debug_pred$1.txt"
mkdir -p "$(dirname "${output}")"
python nmt_transformer_infer.py \
    --model=/home/lab/NeMo/examples/nlp/machine_translation/ngc_results/aayn_base_max_punc/2308718/nemo_experiments/AAYNBase/2021-09-20_09-43-54/checkpoints/AAYNBase.nemo \
    --srctext="${ds_path}/test/debug_input$1.txt" \
    --tgtout="${output}" \
    --target_lang en \
    --source_lang en \

