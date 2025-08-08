#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# NOTE: please check README.md for data preparation scripts
# train_set=train_v3
# valid_set=dev_v3
# test_sets=dev_v3
train_set=tr_no_dev
valid_set=dev
test_sets=eval1

# nbpe=50000
nbpe=5000
s2t_config=conf/train_s2t_multitask-ctc_ebf27_conv2d8_size1024.yaml
inference_config=conf/decode_s2t.yaml

./s2t_ctc.sh \
    --use_lm false \
    --num_nodes 16 \
    --ngpu 4 \
    --nj 64 \
    --gpu_inference true \
    --inference_nj 8 \
    --num_splits_s2t 12 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_input_sentence_size 15000000 \
    --s2t_config "${s2t_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "dump/raw/${train_set}/text" \
    --bpe_nlsyms data/nlsyms.txt \
    --lm_train_text "dump/raw/${train_set}/text" \
    --s2t_args "--use_wandb true --wandb_project owsm_ctc_v3.1 --wandb_entity owsm_ctc_v3.1_test" \
    --lm_args "--use_wandb true --wandb_project owsm_ctc_v3.1 --wandb_entity owsm_ctc_v3.1_test" "$@"
