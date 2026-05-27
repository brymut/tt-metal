# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the HiFi-GAN vocoder for CosyVoice.

Converts mel spectrograms (80-dim) to audio waveform (22050/24000 Hz).

For Stage 1 bring-up, this uses the PyTorch reference implementation
as a CPU fallback. ConvTranspose1d (required for HiFi-GAN upsample blocks)
does not have native TTNN support.

Phase 2 will migrate Conv1d-based components (ResBlocks) to TTNN while
keeping ConvTranspose1d on CPU, or implement via ttnn.conv_transpose2d
with 1D→2D reshaping.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TtHiFiGAN(torch.nn.Module):
    """HiFi-GAN vocoder wrapper for CosyVoice TTNN pipeline.

    Stage 1: Full CPU fallback using PyTorch reference weights.
    This is acceptable because:
    1. HiFi-GAN is ~14M params (small relative to LLM)
    2. Vocoder is called once per chunk, not autoregressively
    3. ConvTranspose1d lacks TTNN support

    The interface matches CosyVoice's hift module for drop-in compatibility.
    """

    def __init__(
        self,
        configs: dict,
        hift_state_dict: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.configs = configs

        in_channels = configs.get("in_channels", 80)
        upsample_rates = configs.get("upsample_rates", [8, 8, 2, 2])
        upsample_kernel_sizes = configs.get("upsample_kernel_sizes", [16, 16, 4, 4])
        upsample_initial_channel = configs.get("upsample_initial_channel", 512)
        resblock_kernel_sizes = configs.get("resblock_kernel_sizes", [3, 7, 11])
        resblock_dilation_sizes = configs.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])

        # Build HiFi-GAN generator on CPU
        self.generator = HiFiGANGenerator(
            in_channels=in_channels,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_initial_channel=upsample_initial_channel,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
        )

        # Load weights
        if hift_state_dict:
            # CosyVoice stores HiFi-GAN weights with 'generator.' prefix
            cleaned_dict = {k.replace("generator.", ""): v for k, v in hift_state_dict.items()}
            try:
                self.generator.load_state_dict(cleaned_dict, strict=False)
            except RuntimeError:
                # Fall back to loading whatever matches
                model_dict = self.generator.state_dict()
                filtered = {k: v for k, v in cleaned_dict.items() if k in model_dict}
                model_dict.update(filtered)
                self.generator.load_state_dict(model_dict)

        self.generator.eval()

    @torch.inference_mode()
    def inference(
        self,
        speech_feat: torch.Tensor,
        cache_source: Optional[torch.Tensor] = None,
        finalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert mel spectrogram to audio waveform.

        Args:
            speech_feat: Mel spectrogram (1, 80, mel_len) or (1, mel_len, 80)
            cache_source: Source cache for streaming (1, 1, cache_len)
            finalize: Whether this is the final chunk

        Returns:
            Tuple of (audio_waveform, source_signal)
            audio: (1, audio_len) where audio_len = mel_len * hop_size
        """
        # Ensure mel is in (B, C, T) format
        if speech_feat.dim() == 3 and speech_feat.shape[1] != 80 and speech_feat.shape[2] == 80:
            speech_feat = speech_feat.transpose(1, 2)

        # Generate audio
        audio = self.generator(speech_feat)

        # Squeeze to (B, T)
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        # Source signal (for streaming cache)
        source = audio.unsqueeze(1)  # (1, 1, T)

        return audio, source


class ResBlock(nn.Module):
    """HiFi-GAN residual block with dilated convolutions."""

    def __init__(self, channels: int, kernel_size: int, dilations: list):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilations:
            self.convs1.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d,
                    padding=(kernel_size * d - d) // 2,
                )
            )
            self.convs2.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = F.leaky_relu(x, 0.1)
            x = conv1(x)
            x = F.leaky_relu(x, 0.1)
            x = conv2(x)
            x = x + residual
        return x


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN generator for waveform reconstruction.

    Architecture:
        conv_pre -> [upsample + resblocks] x N -> conv_post -> tanh
    """

    def __init__(
        self,
        in_channels: int = 80,
        upsample_rates: list = None,
        upsample_kernel_sizes: list = None,
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: list = None,
        resblock_dilation_sizes: list = None,
    ):
        super().__init__()

        if upsample_rates is None:
            upsample_rates = [8, 8, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 16, 4, 4]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        # Initial convolution
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)

        # Upsample blocks
        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    ch,
                    ch // 2,
                    kernel,
                    stride=rate,
                    padding=(kernel - rate) // 2,
                )
            )
            ch = ch // 2

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch_i = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch_i, k, d))

        # Final convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate audio from mel spectrogram.

        Args:
            x: Mel spectrogram (B, 80, T)

        Returns:
            Audio waveform (B, 1, T * prod(upsample_rates))
        """
        x = self.conv_pre(x)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            # Apply residual blocks for this upsample level
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j]
                if xs is None:
                    xs = rb(x)
                else:
                    xs = xs + rb(x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x
