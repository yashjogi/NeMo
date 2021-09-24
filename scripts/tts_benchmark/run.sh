#!/usr/bin/env bash

#  --mixer-ckpt-path=/home/stanislavv/data/ckpts/nemo/mixer-tts/lj_new_mixer_tts_aGF7NdKLwFdsk--val_mel_loss=0.5616-epoch=999-last_ptl_fix.ckpt \
#  --mixer-ckpt-path=/home/stanislavv/data/ckpts/nemo/mixer-tts/lj_new_mixer_tts_ufcsJbLKhWPF8--val_mel_loss=0.5541-epoch=999-last_ptl_fix.ckpt \
#  --mixer-ckpt-path=/home/stanislavv/data/ckpts/nemo/mixer-tts/lj_new_mixer_tts_v7biccFf2Z5pR--val_mel_loss=0.5986-epoch=999-last_ptl_fix.ckpt \
python scripts/tts_benchmark/inference.py \
  --mixer-ckpt-path=/home/stanislavv/data/ckpts/nemo/mixer-tts/lj_new_mixer_tts_7MUAttWjw2zhr--val_mel_loss=0.5424-epoch=999-last_ptl_fix.ckpt \
  --without-matching \
  --manifest-path=/home/stanislavv/data/sets/ljspeech/local/nvidia-split/test.jsonl \
  --cudnn-benchmark --n-repeats=1 --batch-size=1 --n-samples=128
