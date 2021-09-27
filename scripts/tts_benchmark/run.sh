#!/usr/bin/env bash

N_REPEATS=${N_REPEATS:-10}  # More is unnecessary.
BATCH_SIZE=${BATCH_SIZE:-1}
N_CHARS=${N_CHARS:-128}
N_SAMPLES=${N_SAMPLES:-1024}

model_name=$1
# shellcheck disable=SC2154
if [[ $model_name -eq "mixer-tts-1" ]]
then
  model_ckpt_path=/home/stanislavv/data/ckpts/nemo/mixer-tts/lj_new_mixer_tts_aGF7NdKLwFdsk--val_mel_loss=0.5616-epoch=999-last_ptl_fix.ckpt
  model_args="--without-matching"
elif [[ $model_name -eq "mixer-tts-2" ]]
then
  model_ckpt_path=/home/stanislavv/data/ckpts/nemo/mixer-tts/lj_new_mixer_tts_ufcsJbLKhWPF8--val_mel_loss=0.5541-epoch=999-last_ptl_fix.ckpt
  model_args="--without-matching"
elif [[ $model_name -eq "mixer-tts-3" ]]
then
  model_ckpt_path=/home/stanislavv/data/ckpts/nemo/mixer-tts/lj_new_mixer_tts_7MUAttWjw2zhr--val_mel_loss=0.5424-epoch=999-last_ptl_fix.ckpt
  model_args=" "
elif [[ $model_name -eq "fastpitch" ]]
then
  model_ckpt_path=/home/stanislavv/data/ckpts/nemo/mixer-tts/lj_new_mixer_tts_v7biccFf2Z5pR--val_mel_loss=0.5986-epoch=999-last_ptl_fix.ckpt
  model_args="--without-matching"
else
  echo "Wrong model name."
  exit 1
fi

python scripts/tts_benchmark/inference.py \
  --model-ckpt-path=$model_ckpt_path \
  --manifest-path=/home/stanislavv/data/sets/ljspeech/local/nvidia-split/test.jsonl \
  --cudnn-benchmark --n-repeats="$N_REPEATS" \
  --batch-size="$BATCH_SIZE" --n-chars="$N_CHARS" --n-samples="$N_SAMPLES" \
  $model_args
