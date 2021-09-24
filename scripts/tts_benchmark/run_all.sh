#!/usr/bin/env bash

MODELS="mixer-tts-1 mixer-tts-2 mixer-tts-3 fastpitch"

result=""
for MODEL in $MODELS ; do
  rtf=$(scripts/tts_benchmark/run.sh "$MODEL" | grep RTF)
  result="$result$MODEL: $rtf\n"
done

echo -ne "$result"
