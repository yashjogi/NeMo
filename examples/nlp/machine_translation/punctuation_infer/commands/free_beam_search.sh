if [ -z "${batch_size}" ]; then
    batch_size=256
fi
ds_path=/home/apeganov/NeMo/examples/speech_translation/prepared_punctuation_data_all_punctuation
output="${ds_path}/predictions/aayn_base_free_beam_search_job2308718/pred${batch_size}${branch}.txt"
mkdir -p "$(dirname "${output}")"
python nmt_transformer_infer.py \
    --model=/home/apeganov/NeMo/examples/nlp/machine_translation/ngc_results/aayn_base_max_punc/2308718/nemo_experiments/AAYNBase/2021-09-20_09-43-54/checkpoints/AAYNBase.nemo \
    --srctext="${ds_path}/test/input.txt" \
    --tgtout="${output}" \
    --target_lang en \
    --source_lang en \
    --batch_size "${batch_size}"
