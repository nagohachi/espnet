#!/usr/bin/env python3
#  2022, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers Decoder."""

import copy
import logging
from typing import Any, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask

try:
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    is_transformers_available = True
except ImportError:
    is_transformers_available = False

from peft import LoraConfig, TaskType, get_peft_model


class HuggingFaceTransformersDecoder(AbsDecoder):
    """Hugging Face Transformers Decoder."""

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        model_name_or_path: str,
        lora_r: Optional[int],
        lora_alpha: Optional[int],
        lora_dropout: Optional[float],
        causal_lm: bool = False,
        prefix: str = "",
        postfix: str = "",
        overriding_architecture_config: dict[str, Any] | str | None = None,
        load_pretrained_weights: bool = True,
        separate_lm_head: bool = False,
    ):
        super().__init__()

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it "
                "via `pip install transformers`."
            )

        self.load_pretrained_weights = load_pretrained_weights
        self.separate_lm_head = separate_lm_head
        self.causal_lm = causal_lm
        self.model_name_or_path = model_name_or_path

        if overriding_architecture_config is None:
            self.overriding_architecture_config: dict[str, Any] = {}
        elif isinstance(overriding_architecture_config, str):
            self.overriding_architecture_config = read_json_config(
                overriding_architecture_config
            )
        else:
            self.overriding_architecture_config = overriding_architecture_config

        if self.causal_lm:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **self.overriding_architecture_config
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path, **self.overriding_architecture_config
            )

        original_vocab_size = model.config.vocab_size
        if vocab_size > original_vocab_size:
            model.resize_token_embeddings(vocab_size)
            logging.info(
                f"Resized embeddings from {original_vocab_size} to {vocab_size}"
            )
            self.output_vocab_size = vocab_size
        else:
            logging.info(
                f"Keeping original vocab_size {original_vocab_size} (requested {vocab_size})"
            )
            self.output_vocab_size = vocab_size

        self.use_peft = False
        if lora_r is not None or lora_alpha is not None or lora_dropout is not None:
            assert (
                lora_r is not None
                and lora_alpha is not None
                and lora_dropout is not None
            )
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            model = get_peft_model(model, peft_config)
            self.use_peft = True
            logging.info("Applied PEFT/LoRA to the model")

        self.llm = model

        if self.causal_lm:
            config = model.config
            if config.pad_token_id is not None and config.pad_token_id != -1:
                self.decoder_pad_token_id = config.pad_token_id
            else:
                self.decoder_pad_token_id = 1

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.tokenizer_padding_side = tokenizer.padding_side

            self.register_buffer(
                "prefix_ids", tokenizer.encode(prefix, return_tensors="pt").long()
            )
            self.register_buffer(
                "postfix_ids", tokenizer.encode(postfix, return_tensors="pt").long()
            )

            hidden_size = config.hidden_size

            # Set self.decoder for causal_lm to enable state_dict loading in inference
            self.decoder = get_hugging_face_model_network(model)
        else:
            # For Seq2Seq models
            if hasattr(model, "model"):
                self.decoder = model.model.decoder
            else:
                self.decoder = model.decoder
            hidden_size = self.decoder.config.hidden_size

        if encoder_output_size != hidden_size:
            self.linear_in = torch.nn.Linear(encoder_output_size, hidden_size)
        else:
            self.linear_in = torch.nn.Identity()

        if self.separate_lm_head:
            self.lm_head = cast(
                torch.nn.Module, copy.deepcopy(get_hugging_face_model_lm_head(model))
            )
        else:
            # Always keep a reference to lm_head for inference compatibility
            self.lm_head = get_hugging_face_model_lm_head(model)

    def _get_embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings by navigating through model hierarchy.

        Handles DDP wrapping by using getattr to navigate through the model.
        """
        model = self.llm
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens(input_ids)
        elif (
            hasattr(model, "model")
            and hasattr(model.model, "model")
            and hasattr(model.model.model, "embed_tokens")
        ):
            return model.model.model.embed_tokens(input_ids)
        elif (
            hasattr(model, "model")
            and hasattr(model.model, "model")
            and hasattr(model.model.model, "model")
            and hasattr(model.model.model.model, "embed_tokens")
        ):
            return model.model.model.model.embed_tokens(input_ids)
        # For GPT-style models
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            return model.transformer.wte(input_ids)
        # For GPT-NeoX style
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "embed_in"):
            return model.gpt_neox.embed_in(input_ids)
        else:
            raise AttributeError(
                f"Cannot find embed_tokens in model structure: {type(model)}"
            )

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad: input tensor (batch, maxlen_out, #mels)
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        enc_out = self.linear_in(hs_pad)

        if self.causal_lm:
            args, no_loss_lengths = self.add_prefix_postfix(
                enc_out, hlens, ys_in_pad, ys_in_lens
            )

            # The model returns logits directly (lm_head is applied)
            outputs = self.llm(**args)
            x = outputs.logits

            if self.tokenizer_padding_side == "left":
                x = torch.vstack(
                    [
                        F.pad(
                            x[i, -ys_in_lens[i] :, :],
                            (0, 0, 0, int(ys_in_lens.max() - ys_in_lens[i])),
                        ).unsqueeze(0)
                        for i in range(x.shape[0])
                    ]
                )
            else:
                x = torch.vstack(
                    [
                        F.pad(
                            x[
                                i,
                                no_loss_lengths[i] : no_loss_lengths[i] + ys_in_lens[i],
                                :,
                            ],
                            (0, 0, 0, int(ys_in_lens.max() - ys_in_lens[i])),
                        ).unsqueeze(0)
                        for i in range(x.shape[0])
                    ]
                )

        else:
            # Seq2Seq model path
            args: dict = {"return_dict": True}

            if self.decoder.__class__.__name__ == "MBartDecoder":
                ys_in_pad[:, 0] = 2

            args["input_ids"] = ys_in_pad
            mask = (~make_pad_mask(ys_in_lens)).to(ys_in_pad.device).float()
            args["attention_mask"] = mask

            args["encoder_hidden_states"] = enc_out
            hs_mask = (~make_pad_mask(hlens)).to(hs_pad.device).float()
            args["encoder_attention_mask"] = hs_mask

            x = self.decoder(**args).last_hidden_state

            if self.lm_head is not None:
                x = self.lm_head(x)
            else:
                lm_head = get_hugging_face_model_lm_head(self.llm)
                x = lm_head(x)

        # Slice output to match token_list vocab_size for loss function
        if x.size(-1) > self.output_vocab_size:
            x = x[..., : self.output_vocab_size]

        return x, ys_in_lens

    def reload_pretrained_parameters(self):
        if self.load_pretrained_weights:
            logging.info(
                "reload_pretrained_parameters called but PEFT model retains pretrained weights"
            )
        else:
            logging.info(
                "Skipping the loading of pretrained Transformer model parameters!"
            )

    def add_prefix_postfix(self, enc_out, hlens, ys_in_pad, ys_in_lens):
        args = {}

        hlens_max = (hlens + ys_in_lens).max()

        prefix_embeds = self._get_embed_tokens(self.prefix_ids.to(enc_out.device))
        postfix_embeds = self._get_embed_tokens(self.postfix_ids.to(enc_out.device))

        enc_out_list = []

        for i in range(len(hlens)):
            target_embeds = self._get_embed_tokens(
                ys_in_pad[i : i + 1, 1 : ys_in_lens[i]].to(enc_out.device)
            )

            enc_out_element = [
                prefix_embeds,
                enc_out[i : i + 1, : hlens[i], :],
                postfix_embeds,
                target_embeds,
            ]

            pad_ids = torch.tensor([[self.decoder_pad_token_id]]).to(enc_out.device)
            padding = self._get_embed_tokens(pad_ids).expand(
                -1, hlens_max - (hlens[i] + ys_in_lens[i]), -1
            )

            if self.tokenizer_padding_side == "left":
                enc_out_element.insert(0, padding)
            else:
                enc_out_element.insert(len(enc_out_element), padding)

            enc_out_list.append(torch.cat(enc_out_element, dim=1))

        args["inputs_embeds"] = torch.vstack(enc_out_list)

        no_loss_lengths = self.prefix_ids.size(1) + hlens + self.postfix_ids.size(1) - 1
        inputs_lengths = no_loss_lengths + ys_in_lens

        hs_mask = (~make_pad_mask(inputs_lengths)).to(enc_out.device).float()

        if self.tokenizer_padding_side == "left":
            args["attention_mask"] = hs_mask.flip([1])
        else:
            args["attention_mask"] = hs_mask

        args["return_dict"] = True

        return args, no_loss_lengths


def get_hugging_face_model_lm_head(model):
    """Get LM head from model, handling PEFT wrapper."""
    # For PEFT models, lm_head is at base_model.model.lm_head
    if hasattr(model, "lm_head"):
        return model.lm_head
    if (
        hasattr(model, "base_model")
        and hasattr(model.base_model, "model")
        and hasattr(model.base_model.model, "lm_head")
    ):
        return model.base_model.model.lm_head
    if hasattr(model, "embed_out"):
        return model.embed_out
    raise AttributeError("Can not find the LM head attribute")


def get_hugging_face_model_network(model):
    """Get transformer network from model, handling PEFT wrapper.

    Returns the underlying transformer model (without lm_head).
    """
    # For Llama-style models: model.model
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model
    # For PEFT-wrapped Llama: model.model.model
    if (
        hasattr(model, "model")
        and hasattr(model.model, "model")
        and hasattr(model.model.model, "embed_tokens")
    ):
        return model.model.model
    # For deeper PEFT wrapping
    if (
        hasattr(model, "model")
        and hasattr(model.model, "model")
        and hasattr(model.model.model, "model")
        and hasattr(model.model.model.model, "embed_tokens")
    ):
        return model.model.model.model
    # For GPT-style models: model.transformer
    if hasattr(model, "transformer"):
        return model.transformer
    # For GPT-NeoX style: model.gpt_neox
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox
    raise AttributeError(
        f"Cannot find transformer network in model structure: {type(model)}"
    )


def read_json_config(conf_path):
    """Read a json model config information."""
    import json

    with open(conf_path, "r", encoding="utf-8") as f:
        logging.info("Reading config file from " + conf_path)
        confs = json.load(f)
    assert isinstance(confs, dict)
    return confs
