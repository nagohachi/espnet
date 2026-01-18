from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from typeguard import typechecked

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder


class FrameStackingMLP2PostEncoder(AbsPostEncoder):
    """Post encoder of 2-layered MLP with frame stacking for downsampling."""

    @typechecked
    def __init__(
        self, input_size: int, output_size: int, mid_size: int, downsample_k: int
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size * downsample_k, mid_size),
            nn.ReLU(inplace=True),
            nn.Linear(mid_size, output_size),
            nn.ReLU(inplace=True),
        )
        self.ds_k = downsample_k
        self._output_size = output_size

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform downsampling by frame stacking then input to MLP.

        Args:
            input (torch.Tensor): input tensor of shape (batch_size, seq_len, hidden_size).
            input_lengths (torch.Tensor): input tensor lengths of shape (batch_size, ).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: output tensor and its lengths
        """
        seq_len = input.size(1)
        cutoff_input = input[:, : (seq_len // self.ds_k * self.ds_k), :]

        ds_input = rearrange(
            cutoff_input,
            "bs (ds_seq_len ds_k) hid_size -> bs ds_seq_len (ds_k hid_size)",
            ds_k=self.ds_k,
        )
        ds_input_lengths = input_lengths // self.ds_k

        ds_input = self.mlp(ds_input)

        return ds_input, ds_input_lengths
