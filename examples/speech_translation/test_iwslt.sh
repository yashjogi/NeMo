<< 'MULTILINE-COMMENT'
Before running this script you have to create conda Python 2 environment 'mwerSegmenter'. Otherwise mwerSegmenter
will not start.

The script does not stop if an error occurs so you have to watch logs.

The script has to be run from directory NeMo/examples/speech_translation

mwerSegmenter can be installed as recommended in IWSLT description or by command
git clone https://github.com/PeganovAnton/mwerSegmenter

Parameters of the script are
  dataset_dir: path to directory with year dataset. Obtained when archive IWSLT-SLT.tst2019.en-de.tgz is unpacked
  asr_model: pretrained NGC name or path to NeMo ASR checkpoint
  punctuation_model: pretrained NGC name of PunctuationCapitalizationModel or path to .nemo file
  translation_model: pretrained NGC name or path to NeMo NMT checkpoint
  output_dir: path to directory where results will be stored
  segmented: whether segment input audio before transcription using markup provided in dataset. 1 - segment,
    0 - do not segment
  mwerSegmenter: whether use mwerSegmenter for BLEU calculation. 1 - use mwerSegmenter, 0 - do not use
  use_nmt_for_punctuation_and_capitalization: whether use NMT model for restoring punctuation and capitalization
    0 - use BERT model.
  no_all_upper_label: If 1, then only 2 capitalization labels U and O are used. If 0, then 'u' is for only first
    character capitalization, 'U' for all characters capitalization, and 'O' for no capitalization. This parameter
    is required only for NMT punctuation and capitalization.
  use_inverse_text_normalization: If 1, then `nemo_text_processing/inverse_text_normalization/run_predict.py` is
    used. If 0, then `test_iwslt_and_perform_all_ops_common_scripts/text_to_numbers.py` is used.
Usage example:
bash test_iwslt.sh ~/data/IWSLT.tst2019 \
  stt_en_citrinet_1024 \
  punctuation_en_nmt \
  ~/checkpoints/wmt21_en_de_backtranslated_24x6_averaged.nemo \
  ~/iwslt_2019_test_result \
  0 \
  1 \
  1 \
  1 \
  0
MULTILINE-COMMENT

set -e


dataset_dir="$(realpath "$1")"
asr_model="$2"  # Path to checkpoint or NGC pretrained name
punctuation_model="$3"  # Path to checkpoint or NGC pretrained name
translation_model="$4"
output_dir="$(realpath "$5")"
segmented="$6"  # 1 or 0
mwerSegmenter="$7"  # 1 or 0
use_nmt_for_punctuation_and_capitalization="$8"  # 1 or 0
no_all_upper_label="$9"  # 1 or 0
use_inverse_text_normalization="${10}"


audio_dir="${dataset_dir}/wavs"
asr_model_name="$(basename "${asr_model}")"
translation_model_name=$(basename "${translation_model}")

en_ground_truth_manifest="${output_dir}/en_ground_truth_manifest.json"


if [ "${asr_model: -5}" = ".nemo" ]; then
  asr_model_argument_name=model_path
else
  asr_model_argument_name=pretrained_name
fi


if [ "${translation_model: -5}" = ".nemo" ]; then
  translation_model_parameter="-p"
else
  translation_model_parameter="-m"
fi


printf "Creating IWSLT manifest..\n"
python test_iwslt_and_perform_all_ops_common_scripts/create_iwslt_manifest.py -a "${audio_dir}" \
  -t "${dataset_dir}/IWSLT.TED.tst2019.en-de.en.xml" \
  -o "${en_ground_truth_manifest}"


if [ "${segmented}" -eq 1 ]; then
  printf "\n\nSplitting audio files..\n"
  split_data_path="${output_dir}/split"
  python test_iwslt_and_perform_all_ops_common_scripts/iwslt_split_audio.py -a "${dataset_dir}/wavs" \
    -s "${dataset_dir}/IWSLT.TED.tst2019.en-de.yaml" \
    -d "${split_data_path}"
  split_transcripts="${dataset_dir}/split_transcripts/${asr_model_name}"
  transcript_no_numbers="${output_dir}/transcripts_segmented_input_no_numbers/${asr_model_name}.manifest"
  mkdir -p "$(dirname "${transcript_no_numbers}")"
  for f in "${split_data_path}"/*; do
    talk_id=$(basename "${f}")
    if [[ "${talk_id}" =~ ^[1-9][0-9]*$ ]]; then
      python ~/NeMo/examples/asr/transcribe_speech.py "${asr_model_argument_name}"="${asr_model}" \
        audio_dir="${f}" \
        output_filename="${split_transcripts}/${talk_id}.manifest" \
        cuda=0 \
        batch_size=4
    fi
  done
  python test_iwslt_and_perform_all_ops_common_scripts/join_split_wav_manifests.py \
    -S "${split_transcripts}" \
    -o "${transcript_no_numbers}" \
    -n "${audio_dir}"
else
  if [ "${segmented}" -ne 0 ]; then
    echo "Wrong value '${segmented}' of fifth parameter of 'translate_and_score.sh'. Only '0' and '1' are supported."
  fi
  transcript_no_numbers="${output_dir}/transcripts_not_segmented_input_no_numbers/${asr_model_name}.manifest"
  mkdir -p "$(dirname "${transcript_no_numbers}")"
  python ~/NeMo/examples/asr/transcribe_speech.py "${asr_model_argument_name}"="${asr_model}" \
    audio_dir="${audio_dir}" \
    output_filename="${transcript_no_numbers}" \
    cuda=0 \
    batch_size=1
fi


printf "\n\nTransforming text to numbers.."
if [ "${segmented}" -eq 1 ]; then
  transcript="${output_dir}/transcripts_segmented_input/${asr_model_name}.manifest"
else
  transcript="${output_dir}/transcripts_not_segmented_input/${asr_model_name}.manifest"
fi
mkdir -p "$(dirname "${transcript}")"
if [ "${use_inverse_text_normalization}" -eq 1 ]; then
  python -m nemo_text_processing.inverse_text_normalization.run_predict.py \
    --input_manifest "${transcript_no_numbers}" \
    --output_manifest "${transcript}" \
    --manifest_text_key pred_text \
    --input_case lower_cased
else
  python test_iwslt_and_perform_all_ops_common_scripts/text_to_numbers.py \
    -i "${transcript_no_numbers}" \
    -o "${transcript}"
fi


printf "\n\nComputing WER..\n"
wer_by_transcript_and_audio="${output_dir}/wer_by_transcript_and_audio"
if [ "${segmented}" -eq 1 ]; then
  wer_dir="segmented"
else
  wer_dir="not_segmented"
fi
wer="$(python iwslt_scoring/wer_between_2_manifests.py "${transcript}" "${en_ground_truth_manifest}" \
      -o "${wer_by_transcript_and_audio}/${wer_dir}/${asr_model_name}.json")"
echo "WER: ${wer}"


printf "\n\nAdding punctuation and restoring capitalization..\n"
if [ "${segmented}" -eq 1 ]; then
  punc_dir="${output_dir}/punc_transcripts_segmented_input"
else
  punc_dir="${output_dir}/punc_transcripts_not_segmented_input"
fi
if [ "${use_nmt_for_punctuation_and_capitalization}" -eq 1 ]; then
  if [ "${no_all_upper_label}" -eq 1 ]; then
    python ../nlp/machine_translation/punctuation_infer/punctuate_capitalize_nmt.py \
      --input_manifest ${transcript} \
      --output_text "${punc_dir}/${asr_model_name}.txt" \
      --model_path "${punctuation_model}" \
      --max_seq_length 128 \
      --step 8 \
      --margin 16 \
      --batch_size 42 \
      --add_source_num_words_to_batch \
      --make_queries_contain_intact_sentences \
      --manifest_to_align_with "${en_ground_truth_manifest}" \
      --no_all_upper_label
  else
    python ../nlp/machine_translation/punctuation_infer/punctuate_capitalize_nmt.py \
      --input_manifest ${transcript} \
      --output_text "${punc_dir}/${asr_model_name}.txt" \
      --model_path "${punctuation_model}" \
      --max_seq_length 128 \
      --step 8 \
      --margin 16 \
      --batch_size 42 \
      --add_source_num_words_to_batch \
      --make_queries_contain_intact_sentences \
      --manifest_to_align_with "${en_ground_truth_manifest}"
  fi
else
  python test_iwslt_and_perform_all_ops_common_scripts/punc_cap.py -a "${en_ground_truth_manifest}" \
    -m "${punctuation_model}" \
    -p "${transcript}" \
    -o "${punc_dir}/${asr_model_name}.txt"
fi


printf "\n\nTranslating..\n"
if [ "${segmented}" -eq 1 ]; then
  translation_dir="${output_dir}/translations_segmented_input"
else
  translation_dir="${output_dir}/translations_not_segmented_input"
fi
translated_text="${translation_dir}/${translation_model_name}/${asr_model_name}.txt"
python test_iwslt_and_perform_all_ops_common_scripts/translate_iwslt.py \
  "${translation_model_parameter}" "${translation_model}" \
  -i "${punc_dir}/${asr_model_name}.txt" \
  -o "${translated_text}" \
  -s


if [ "${mwerSegmenter}" -eq 1 ]; then
  printf "\n\nSegmenting translations using mwerSegmenter..\n"
  if [ "${segmented}" -eq 1 ]; then
    translation_dir_mwer_xml="${output_dir}/mwer_translations_xml_segmented_input"
    translation_dir_mwer_txt="${output_dir}/mwer_translations_txt_segmented_input"
  else
    translation_dir_mwer_xml="${output_dir}/mwer_translations_xml_not_segmented_input"
    translation_dir_mwer_txt="${output_dir}/mwer_translations_txt_not_segmented_input"
  fi
  translated_mwer_xml="${translation_dir_mwer_xml}/${translation_model_name}/${asr_model_name}.xml"
  mkdir -p "$(dirname "${translated_mwer_xml}")"
  translated_text_for_scoring="${translation_dir_mwer_txt}/${translation_model_name}/${asr_model_name}.txt"
  (
  __conda_setup="$("/home/${USER}/anaconda3/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
  if [ $? -eq 0 ]; then
    eval "$__conda_setup"
  else
    if [ -f "/home/${USER}/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/${USER}/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/${USER}/anaconda3/bin:$PATH"
    fi
  fi
  unset __conda_setup
  set -e
  conda activate mwerSegmenter  # python 2 conda environment
  cd ~/mwerSegmenter/
  ./segmentBasedOnMWER.sh "${dataset_dir}/IWSLT.TED.tst2019.en-de.en.xml" \
    "${dataset_dir}/IWSLT.TED.tst2019.en-de.de.xml" \
    "${translated_text}" \
    "${asr_model_name}" \
    German \
    "${translated_mwer_xml}" \
    no \
    1
  conda deactivate
  )
  reference="${output_dir}/iwslt_de_text_by_segs.txt"
  python test_iwslt_and_perform_all_ops_common_scripts/xml_2_text_segs_2_lines.py \
    -i "${dataset_dir}/IWSLT.TED.tst2019.en-de.de.xml" \
    -o "${reference}"
  mkdir -p "$(dirname "${translated_text_for_scoring}")"
  python test_iwslt_and_perform_all_ops_common_scripts/xml_2_text_segs_2_lines.py \
    -i "${translated_mwer_xml}" \
    -o "${translated_text_for_scoring}"
else
  if [ "${segmented}" -ne 0 ]; then
    echo "Wrong value '${mwerSegmenter}' of sixth parameter of 'translate_and_score.sh'. Only '0' and '1' are supported."
  fi
  translated_text_for_scoring="${translated_text}"
  reference="${output_dir}/iwslt_de_text_by_wavs.txt"
  python test_iwslt_and_perform_all_ops_common_scripts/prepare_iwslt_text_for_translation.py \
    -a "${en_ground_truth_manifest}" \
    -t "${dataset_dir}/IWSLT.TED.tst2019.en-de.de.xml" \
    -o "${reference}" \
    -j
fi


printf "\n\nComputing BLEU..\n"
bleu=$(sacrebleu "${reference}" -i "${translated_text_for_scoring}" -m bleu -b -w 4)
echo "BLEU: ${bleu}"
if [ "${segmented}" -eq 1 ]; then
  output_file="${output_dir}/BLEU_segmented_input/${translation_model_name}/${asr_model_name}.txt"
else
  output_file="${output_dir}/BLEU_not_segmented_input/${translation_model_name}/${asr_model_name}.txt"
fi
mkdir -p "$(dirname "${output_file}")"
echo "" >> "${output_file}"
echo "$(date)" >> "${output_file}"
echo "segmented: ${segmented}" >> "${output_file}"
echo "mwerSegmenter: ${mwerSegmenter}" >> "${output_file}"
echo "ASR model: ${asr_model}" >> "${output_file}"
echo "NMT model: ${translation_model}" >> "${output_file}"
echo "BLUE: ${bleu}" >> "${output_file}"


if [ "${mwerSegmenter}" -eq 1 ]; then
  printf "\n\nComputing BLEU for separate talks..\n"
  if [ "${segmented}" -eq 1 ]; then
    separate_files_translations="${output_dir}/translations_for_docs_in_separate_files_segmented_input"
    bleu_separate_files="${output_dir}/BLUE_by_docs_segmented_input"
  else
    separate_files_translations="${output_dir}/translations_for_docs_in_separate_files_not_segmented_input"
    bleu_separate_files="${output_dir}/BLUE_by_docs_not_segmented_input"
  fi
  bash iwslt_scoring/compute_bleu_for_separate_talks_one_model.sh \
    "${translated_mwer_xml}" \
    "${dataset_dir}/IWSLT.TED.tst2019.en-de.de.xml" \
    "${separate_files_translations}/${translation_model_name}/${asr_model_name}" \
    "${output_dir}/reference_for_docs_in_separate_files" \
    "${bleu_separate_files}/${translation_model_name}/${asr_model_name}.txt"
fi

set +e