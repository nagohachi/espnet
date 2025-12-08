import logging
import os
from contextlib import nullcontext
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import WavLMModel
from transformers.models.wavlm.modeling_wavlm import WavLMBaseModelOutput
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder

SUPPORTED_MODELS = ("wavlm-large", "wavlm-base-plus")


def make_pad_mask(ilens: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Create padding mask.

    Args:
        ilens: Sequence lengths (batch_size,)
        max_len: Maximum length. If None, uses max of ilens.

    Returns:
        Padding mask (batch_size, max_len) where True indicates padding.
    """
    if max_len is None:
        max_len = int(ilens.max().item())

    indices = torch.arange(max_len, device=ilens.device, dtype=ilens.dtype).unsqueeze(0)
    return indices >= ilens.unsqueeze(1)


class WavLMEncoder(AbsEncoder):
    @typechecked
    def __init__(
        self,
        input_size: int,
        base_model: Literal["microsoft/wavlm-large", "microsoft/wavlm-base-plus"] = "microsoft/wavlm-large",
        freeze_bottom_layers: int = 0,
        freeze_feature_encoder: bool = True,
        gradual_unfreeze_steps: int = 0,
        initial_freeze_bottom_layers: int = -1,
        encoder_lr: Optional[float] = None,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        feat_proj_dropout: float = 0.0,
        mask_time_prob: float = 0.05,
        mask_time_length: int = 10,
        mask_feature_prob: float = 0.0,
        mask_feature_length: int = 10,
        layerdrop: float = 0.0,
    ) -> None:
        """WavLMEncoder class.

        Args:
            input_size: The number of expected features in the input (unused, for interface compatibility).
            base_model: The base model to use.
            freeze_bottom_layers: Number of bottom layers to freeze. -1 freezes all layers.
            freeze_feature_encoder: Whether to freeze the feature encoder.
            gradual_unfreeze_steps: If > 0, apply initial_freeze_bottom_layers until this step,
                then switch to freeze_bottom_layers setting.
            initial_freeze_bottom_layers: Number of bottom layers to freeze before gradual_unfreeze_steps.
                -1 (default) freezes all layers. Only used when gradual_unfreeze_steps > 0.
            encoder_lr: Learning rate for encoder parameters. If specified, sets _optim attribute
                on encoder parameters for use with configure_optimizer (requires exclude_weight_decay=true).
            interctc_layer_idx: Indices of encoder layers to apply intermediate CTC.
                Layer indices are 1-indexed (e.g., [6, 12] for 6th and 12th layers).
            interctc_use_conditioning: Whether to use CTC output for conditioning.
                When True, CTC softmax output is projected and added to hidden states.
        """
        super().__init__()
        self.base_model = base_model
        self.freeze_bottom_layers = freeze_bottom_layers
        self.freeze_feature_encoder = freeze_feature_encoder
        self.gradual_unfreeze_steps = gradual_unfreeze_steps
        self.initial_freeze_bottom_layers = initial_freeze_bottom_layers
        self._is_unfrozen = False
        self.encoder_lr = encoder_lr
        self.interctc_layer_idx = interctc_layer_idx
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = (
            None  # Set by ESPnetASRModel if interctc_use_conditioning=True
        )

        if base_model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Invalid base model: {base_model}. Supported: {SUPPORTED_MODELS}"
            )

        self.model = WavLMModel.from_pretrained(
            base_model,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            feat_proj_dropout=feat_proj_dropout,
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length,
            mask_feature_prob=mask_feature_prob,
            mask_feature_length=mask_feature_length,
            layerdrop=layerdrop,
        )

        # Validate interctc_layer_idx
        num_layers = len(self.model.encoder.layers)
        if len(interctc_layer_idx) > 0:
            if min(interctc_layer_idx) < 1 or max(interctc_layer_idx) > num_layers:
                raise ValueError(
                    f"interctc_layer_idx values must be between 1 and {num_layers}, "
                    f"got {interctc_layer_idx}"
                )

        if gradual_unfreeze_steps > 0:
            self._configure_initial_freezing()
        else:
            self._configure_freezing()
            self._is_unfrozen = True
        self._log_trainable_params()

        # Set _optim attribute for encoder-specific learning rate
        if encoder_lr is not None:
            self._set_encoder_lr(encoder_lr)

        if len(self.interctc_layer_idx):
            logging.info(
                f"Using intermediate ctc with conditioning={self.interctc_use_conditioning}"
            )

    def _configure_initial_freezing(self) -> None:
        """Configure initial freezing based on initial_freeze_bottom_layers."""
        if self.initial_freeze_bottom_layers == -1:
            # Freeze all WavLM layers (only CTC head trains)
            self._set_requires_grad(self.model, False)
        else:
            # Freeze bottom layers, unfreeze top layers
            self._set_requires_grad(self.model, False)
            for layer in self.model.encoder.layers[self.initial_freeze_bottom_layers :]:
                self._set_requires_grad(layer, True)

    def _configure_freezing(self) -> None:
        """Configure which parameters to freeze based on settings."""
        if self.freeze_bottom_layers == -1:
            # Freeze all layers
            self._set_requires_grad(self.model, False)
            return

        # Freeze all, then selectively unfreeze
        self._set_requires_grad(self.model, False)

        # Unfreeze layers above freeze_bottom_layers
        for layer in self.model.encoder.layers[self.freeze_bottom_layers :]:
            self._set_requires_grad(layer, True)

        # Unfreeze encoder components when freeze_bottom_layers == 0
        if self.freeze_bottom_layers == 0:
            self._set_requires_grad(self.model.feature_projection, True)
            self._set_requires_grad(self.model.encoder.pos_conv_embed, True)
            self._set_requires_grad(self.model.encoder.layer_norm, True)

        # Handle feature encoder freezing
        if self.freeze_feature_encoder:
            self._set_requires_grad(self.model.feature_extractor, False)
            if self.freeze_bottom_layers > 0:
                self._set_requires_grad(self.model.feature_projection, True)

    def maybe_unfreeze(self, step: int) -> bool:
        """Unfreeze layers if step threshold is reached.

        Args:
            step: Current training step.

        Returns:
            True if unfreezing occurred at this step, False otherwise.
        """
        if self._is_unfrozen or self.gradual_unfreeze_steps <= 0:
            return False

        if step >= self.gradual_unfreeze_steps:
            self._configure_freezing()
            self._is_unfrozen = True
            self._log_trainable_params(prefix=f"[Unfreeze at step {step}] ")
            return True
        return False

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        """Set requires_grad for all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _set_encoder_lr(self, lr: float) -> None:
        """Set _optim attribute on encoder parameters for custom learning rate.

        This works with configure_optimizer() in espnet2/optimizers/optim_groups.py
        to create separate param groups with different learning rates.
        Requires exclude_weight_decay=true in training config.
        """
        for param in self.model.parameters():
            if param.requires_grad:
                setattr(param, "_optim", {"lr": lr})

    def _log_trainable_params(self, prefix: str = "") -> None:
        """Log trainable parameters (only on rank 0)."""
        if os.environ.get("LOCAL_RANK", "0") != "0":
            return

        components = [
            ("feature_extractor", self.model.feature_extractor),
            ("feature_projection", self.model.feature_projection),
            ("encoder", self.model.encoder),
        ]

        for name, module in components:
            params, percentage = self._count_trainable_parameters(module)
            logging.info(
                f"{prefix}Trainable parameters of {name}: {params:.2f}M ({percentage:.2f}%)"
            )

    @staticmethod
    def _count_trainable_parameters(module: nn.Module) -> tuple[float, float]:
        """Count trainable parameters and percentage."""
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        percentage = (trainable / total * 100) if total > 0 else 0.0
        return trainable * 1e-6, percentage

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[torch.Tensor] = None,
        ctc: Optional[CTC] = None,
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Forward pass with optional intermediate CTC.

        Args:
            xs_pad: Input tensor (batch, time).
            ilens: Input lengths (batch,).
            prev_states: Not used.
            ctc: CTC module for intermediate CTC conditioning.

        Returns:
            If interctc_layer_idx is empty:
                hidden_states: (batch, time, hidden_size)
            Else:
                (hidden_states, intermediate_outs): where intermediate_outs is
                    a list of (layer_idx, hidden_states) tuples.
            olens: Output lengths (batch,).
            None: Placeholder for states.
        """
        attention_mask = ~make_pad_mask(ilens).to(xs_pad.device)

        # Use inference_mode only when all layers are frozen
        if not self._is_unfrozen:
            use_inference_mode = self.initial_freeze_bottom_layers == -1
        else:
            use_inference_mode = self.freeze_bottom_layers == -1

        if len(self.interctc_layer_idx) == 0:
            # No intermediate CTC: use standard forward
            return self._forward_simple(
                xs_pad, attention_mask, ilens, use_inference_mode
            )
        elif self.interctc_use_conditioning:
            # With conditioning: process layers individually
            return self._forward_with_conditioning(
                xs_pad, attention_mask, ilens, use_inference_mode, ctc
            )
        else:
            # Without conditioning: use standard forward and extract intermediate outputs
            return self._forward_with_interctc(
                xs_pad, attention_mask, ilens, use_inference_mode
            )

    def _forward_simple(
        self,
        xs_pad: torch.Tensor,
        attention_mask: torch.Tensor,
        ilens: torch.Tensor,
        use_inference_mode: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Standard forward pass without intermediate CTC."""
        context = torch.inference_mode() if use_inference_mode else nullcontext()
        with context:
            enc_outputs: WavLMBaseModelOutput = self.model(
                input_values=xs_pad,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = enc_outputs.last_hidden_state
        seq_len = hidden_states.size(1)
        output_lengths = self.model._get_feat_extract_output_lengths(ilens)
        olens = output_lengths.clamp(max=seq_len)

        return hidden_states, olens, None

    def _forward_with_interctc(
        self,
        xs_pad: torch.Tensor,
        attention_mask: torch.Tensor,
        ilens: torch.Tensor,
        use_inference_mode: bool,
    ) -> Tuple[
        Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Forward pass with intermediate CTC outputs (no conditioning)."""
        context = torch.inference_mode() if use_inference_mode else nullcontext()
        with context:
            enc_outputs: WavLMBaseModelOutput = self.model(
                input_values=xs_pad,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # enc_outputs.hidden_states: tuple of (num_layers + 1) tensors
        # hidden_states[0]: after feature projection (before encoder layers)
        # hidden_states[i]: after i-th encoder layer (1-indexed)
        all_hidden_states = enc_outputs.hidden_states
        final_hidden_states = enc_outputs.last_hidden_state
        seq_len = final_hidden_states.size(1)
        output_lengths = self.model._get_feat_extract_output_lengths(ilens)
        olens = output_lengths.clamp(max=seq_len)

        # Collect intermediate outputs at specified layers
        intermediate_outs = []
        for layer_idx in self.interctc_layer_idx:
            # layer_idx is 1-indexed, all_hidden_states[layer_idx] is the output after layer_idx
            intermediate_outs.append((layer_idx, all_hidden_states[layer_idx]))

        return (final_hidden_states, intermediate_outs), olens, None

    def _forward_with_conditioning(
        self,
        xs_pad: torch.Tensor,
        attention_mask: torch.Tensor,
        ilens: torch.Tensor,
        use_inference_mode: bool,
        ctc: Optional[CTC],
    ) -> Tuple[
        Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Forward pass with intermediate CTC conditioning.

        This processes each encoder layer individually to apply CTC conditioning
        at specified layers.
        """
        # When conditioning is enabled, we need gradients for conditioning_layer
        # So we disable inference_mode if conditioning_layer exists
        if self.conditioning_layer is not None:
            use_inference_mode = False
        context = torch.inference_mode() if use_inference_mode else nullcontext()

        with context:
            # Step 1: Feature extraction
            extract_features = self.model.feature_extractor(xs_pad)
            extract_features = extract_features.transpose(1, 2)

            # Step 2: Compute attention mask for encoder
            if attention_mask is not None:
                attention_mask_enc = self.model._get_feature_vector_attention_mask(
                    extract_features.shape[1], attention_mask
                )
            else:
                attention_mask_enc = None

            # Step 3: Feature projection
            hidden_states, _ = self.model.feature_projection(extract_features)

            # Step 4: Prepare attention mask for encoder layers
            # WavLM masks hidden_states directly at padding positions
            if attention_mask_enc is not None:
                # Mask hidden states at padding positions (set to 0)
                expand_attention_mask = attention_mask_enc.unsqueeze(-1).repeat(
                    1, 1, hidden_states.shape[2]
                )
                hidden_states[~expand_attention_mask] = 0
                # Pass 2D attention_mask to layers (same as original WavLMEncoder)
                # attention_mask_enc: True = valid, passed as-is to WavLMEncoderLayer
                layer_attention_mask = attention_mask_enc
            else:
                layer_attention_mask = None

            # Step 5: Position embedding
            position_embeddings = self.model.encoder.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings
            hidden_states = self.model.encoder.layer_norm(hidden_states)
            hidden_states = self.model.encoder.dropout(hidden_states)

            # Step 6: Process each encoder layer
            # Note: position_bias is computed in layer 0 (which has rel_attn_embed)
            # and passed to subsequent layers
            intermediate_outs = []
            position_bias = None

            for layer_idx, layer in enumerate(self.model.encoder.layers):
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=layer_attention_mask,
                    position_bias=position_bias,
                    output_attentions=False,
                    index=layer_idx,  # Required for WavLM relative position bias
                )
                # layer_outputs = (hidden_states, position_bias) or (hidden_states, position_bias, attn_weights)
                hidden_states = layer_outputs[0]
                position_bias = layer_outputs[1]

                # Check if this layer is an interctc layer (1-indexed)
                if (layer_idx + 1) in self.interctc_layer_idx:
                    intermediate_outs.append((layer_idx + 1, hidden_states))

                    # Apply CTC conditioning if enabled
                    if self.conditioning_layer is not None and ctc is not None:
                        ctc_out = ctc.softmax(hidden_states)
                        hidden_states = hidden_states + self.conditioning_layer(ctc_out)

        seq_len = hidden_states.size(1)
        output_lengths = self.model._get_feat_extract_output_lengths(ilens)
        olens = output_lengths.clamp(max=seq_len)

        return (hidden_states, intermediate_outs), olens, None

    def output_size(self) -> int:
        return self.model.config.hidden_size
