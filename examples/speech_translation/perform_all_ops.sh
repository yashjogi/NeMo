set -e
if [ -z "${workdir}" ]; then
  workdir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019
fi

printf "Creating IWSLT manifest.."
python test_iwslt_and_perform_all_ops_common_scripts/create_iwslt_manifest.py -a "${workdir}/wavs" \
  -t "${workdir}/IWSLT.TED.tst2019.en-de.en.xml" \
  -o "${workdir}/manifest.json"

printf "\n\nSplitting audio files..\n"
python test_iwslt_and_perform_all_ops_common_scripts/iwslt_split_audio.py -a "${workdir}/wavs" \
  -s "${workdir}/IWSLT.TED.tst2019.en-de.yaml" \
  -d "${workdir}/split"

printf "\n\nTranscription..\n"
workdir="${workdir}" bash perform_all_ops_only_scripts/transcribe_with_different_models.sh

printf "\n\nComputing WER..\n"
workdir="${workdir}" bash iwslt_scoring/compute_all_wers.sh

printf "\n\nPunctuation and capitalization..\n"
workdir="${workdir}" bash perform_all_ops_only_scripts/punc_cap_all.sh

printf "\n\nTranslation..\n"
workdir="${workdir}" bash perform_all_ops_only_scripts/translate.sh

printf "\n\nCreating de ground truth..\n"
workdir="${workdir}" bash perform_all_ops_only_scripts/create_de_ground_truth.sh

printf "\n\nmwerSegmenting..\n"
workdir="${workdir}" bash perform_all_ops_only_scripts/mwerSegmenter.sh

printf "\n\nPreparing mwer segments for BLEU scoring..\n"
workdir="${workdir}" bash perform_all_ops_only_scripts/prepare_translations_and_references_for_mwer_scoring.sh

printf "\n\nScoring translations..\n"
workdir="${workdir}" bash iwslt_scoring/score_translations.sh

set +e