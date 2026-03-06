#!/usr/bin/env python3
"""Run inference on dev_other with trained LLM ASR model."""

import sys

sys.path.insert(0, "../../../")

from pathlib import Path

import editdistance
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from espnet2.tasks.asr import ASRTask

# Paths
model_path = "exp/asr_wavlm_llama3b_libri_0125_1423/2epoch.pth"
config_path = "exp/asr_wavlm_llama3b_libri_0125_1423/config.yaml"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if device == "cpu":
    print("WARNING: Running on CPU will be very slow and may run out of memory!")
    print("Please run on a GPU node.")

# Build model from config
print("Building model...")
model, train_args = ASRTask.build_model_from_file(config_path, model_path, device)
model.eval()
print("Model loaded successfully")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Load test_other data
wav_scp_path = "dump/raw/test_other/wav.scp"
text_path = "dump/raw/test_other/text"

wav_dict = {}
with open(wav_scp_path) as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            wav_dict[parts[0]] = parts[1]

text_dict = {}
with open(text_path) as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            text_dict[parts[0]] = parts[1]

print(f"Loaded {len(wav_dict)} wav files, {len(text_dict)} texts")

# Test on first N samples
n_test = 5000
utt_ids = list(wav_dict.keys())[:n_test]

all_hyps = []
all_refs = []

# Output file for results
output_file = open("inference_results.txt", "w")

print(f"\nRunning inference on {n_test} samples...")
with torch.no_grad():
    for utt_id in tqdm(utt_ids):
        wav_path = wav_dict[utt_id]
        ref_text = text_dict[utt_id]

        # Load audio
        audio, sr = sf.read(wav_path)

        # Convert to tensor and add batch dimension
        speech = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        speech_lengths = torch.tensor([speech.shape[1]], device=device)

        # Encode (includes postencoder)
        encoder_out, encoder_out_lens = model.encode(speech, speech_lengths)

        # Prepare for LLM decoder
        enc_out = model.decoder.linear_in(encoder_out)

        # Build prefix embeddings
        prefix_embeds = model.decoder._get_embed_tokens(
            model.decoder.prefix_ids.to(device)
        )
        postfix_embeds = model.decoder._get_embed_tokens(
            model.decoder.postfix_ids.to(device)
        )

        # Concat: prefix + encoder_out + postfix
        inputs_embeds = torch.cat([prefix_embeds, enc_out, postfix_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)

        # Generate with LLM
        # Llama 3 EOS tokens (new model trained with sym_eos=<|eot_id|>=128009)
        eos_token_ids = [128001, 128008, 128009]
        outputs = model.decoder.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=256,
            num_beams=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_token_ids,
        )

        # Decode output
        hyp_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Debug: print raw token IDs for first sample
        if len(all_hyps) == 0:
            print(f"Raw output tokens: {outputs[0].tolist()}")
            print(f"Last 10 tokens: {outputs[0][-10:].tolist()}")

        all_hyps.append(hyp_text.lower())
        all_refs.append(ref_text.lower())

        # Write to file and flush
        output_file.write(f"[{utt_id}]\n")
        output_file.write(f"REF: {ref_text}\n")
        output_file.write(f"HYP: {hyp_text}\n\n")
        output_file.flush()

        if len(all_hyps) <= 5:
            print(f"\n[{utt_id}]")
            print(f"REF: {ref_text}")
            print(f"HYP: {hyp_text}")

# Calculate WER
print("\n" + "=" * 50)
print("Results:")
total_words = 0
total_errors = 0
for hyp, ref in zip(all_hyps, all_refs):
    ref_words = ref.split()
    hyp_words = hyp.split()
    errors = editdistance.eval(ref_words, hyp_words)
    total_errors += errors
    total_words += len(ref_words)

wer = total_errors / total_words if total_words > 0 else 0
print(f"WER: {wer * 100:.2f}% ({total_errors}/{total_words})")

# Calculate CER
total_chars = 0
total_char_errors = 0
for hyp, ref in zip(all_hyps, all_refs):
    ref_chars = ref.replace(" ", "")
    hyp_chars = hyp.replace(" ", "")
    errors = editdistance.eval(ref_chars, hyp_chars)
    total_char_errors += errors
    total_chars += len(ref_chars)

cer = total_char_errors / total_chars if total_chars > 0 else 0
print(f"CER: {cer * 100:.2f}% ({total_char_errors}/{total_chars})")

# Write final results to file
output_file.write("=" * 50 + "\n")
output_file.write(f"WER: {wer * 100:.2f}% ({total_errors}/{total_words})\n")
output_file.write(f"CER: {cer * 100:.2f}% ({total_char_errors}/{total_chars})\n")
output_file.close()
print(f"\nResults saved to inference_results.txt")
