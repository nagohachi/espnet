#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/train_wavlm_large_ctc.yaml
inference_config=conf/tuning/decode_ctc_bs1.yaml

current_time=$(date +"%m%d_%H%M")
asr_args=(
    "--num_workers 6"
    "--log_interval 100"
    "--use_wandb true"
    "--wandb_project audio-encoder-pretrain-gridsearch"
    "--wandb_entity chouon-llm"
    "--wandb_name wavlm_large_ctc_quick_libri_100_${current_time}"
    "--use_amp true"
)


./asr.sh \
    --lang en \
    --stage 11 \
    --stop_stage 11 \
    --nj 16 \
    --feats_normalize null \
    --inference_nj 12 \
    --token_type char \
    --max_wav_duration 30 \
    --ngpu 4 \
    --asr_tag wavlm_large_ctc_quick_libri_100_${current_time} \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --asr_args "${asr_args[*]}" \
    --inference_asr_model valid.wer_ctc.best.pth \
    --lm_train_text "data/${train_set}/text" \
     "$@"
