#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/llm_asr/train_asr_wavlm-large_llama-3.2-3b-it.yaml
inference_config=conf/decode_asr.yaml

# Llama 3.2 tokenizer
token_type=hugging_face
hugging_face_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"

./asr.sh \
    --lang en \
    --ngpu 2 \
    --max_wav_duration 30 \
    --token_type "${token_type}" \
    --hugging_face_model_name_or_path "${hugging_face_model_name_or_path}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --use_lm false \
    --cleaner none \
    --speed_perturb_factors "0.9 1.0 1.1" \
    "$@"
