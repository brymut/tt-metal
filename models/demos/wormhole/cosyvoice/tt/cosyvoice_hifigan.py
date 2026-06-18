# SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN native port of the CosyVoice HiFi-GAN vocoder (HiFTGenerator).

Reference: `cosyvoice.hifigan.generator.HiFTGenerator` (non-causal variant).
Config (from `pretrained_models/CosyVoice-300M/cosyvoice.yaml`):
    in_channels=80, base_channels=512, nb_harmonics=8,
    sampling_rate=22050, upsample_rates=[8,8], upsample_kernel_sizes=[16,16],
    istft_params={n_fft: 16, hop_len: 4},
    resblock_kernel_sizes=[3,7,11], resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]],
    source_resblock_kernel_sizes=[7,11], source_resblock_dilation_sizes=[[1,3,5],[1,3,5]],
    lrelu_slope=0.1, audio_limit=0.99, f0_predictor=ConvRNNF0Predictor.

Components:
- `TtF0Predictor`: 5x Conv1d + ELU + Linear (classifier). All on device.
- `TtSnake`: elementwise activation `x + (1/a) * sin(x*a)^2`. On device.
- `TtResBlock1d`: Snake + Conv1d(dilated) + Snake + Conv1d + residual. On device.
- `TtHiFTGenerator`: conv_pre, ups, source_downs, source_resblocks, resblocks,
  conv_post. All on device except `m_source` (random/sine gen, CPU) and STFT/iSTFT
  (CPU).
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn
from models.demos.wormhole.cosyvoice.tt.cosyvoice_unet import TtConv1d


def _q(base_address: str, suffix: str) -> str:
    return f"{base_address}.{suffix}" if base_address else suffix


def deparametrize_weight_norm(state_dict: dict) -> dict:
    """Convert `parametrizations.weight.original0/1` keys to plain `weight` keys.

    PyTorch's `weight_norm` stores the parametrized components as
    `weight.parametrizations.weight.original0` (g, shape `[out, 1, 1]`) and
    `weight.parametrizations.weight.original1` (v, shape `[out, in, k]`). The
    effective weight is `g * v / ||v||` (norm over the trailing (in, k) dims).
    """
    out = dict(state_dict)
    g_keys = [k for k in out if k.endswith(".parametrizations.weight.original0")]
    for gk in g_keys:
        suffix = gk[: -len(".parametrizations.weight.original0")]
        vk = suffix + ".parametrizations.weight.original1"
        if vk not in out:
            continue
        g = out[gk].float()
        v = out[vk].float()
        v_norm = v.norm(dim=list(range(1, v.ndim)), keepdim=True)
        w = g * (v / (v_norm + 1e-12))
        out[suffix + ".weight"] = w.to(state_dict[gk].dtype)
        del out[gk]
        del out[vk]
    return out


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class TtCpuConv1d(nn.Module):
    """CPU-fallback Conv1d for shapes that ttnn.conv1d cannot handle (e.g. in_channels < 32).

    Downloads input to host, runs PyTorch Conv1d, uploads result back to device.
    Keeps the same forward signature as TtConv1d: (x_tt, batch_size, input_length).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        # PyTorch Conv1d weight layout: [out_channels, in_channels, kernel_size]
        self.conv = torch.nn.Conv1d(
            in_channels=weight.shape[1],
            out_channels=weight.shape[0],
            kernel_size=weight.shape[2],
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        with torch.no_grad():
            self.conv.weight.copy_(weight)
            if bias is not None:
                self.conv.bias.copy_(bias)
            else:
                self.conv.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channels = weight.shape[1]
        self.out_channels = weight.shape[0]
        self.kernel_size = weight.shape[2]

    def forward(self, x_tt: ttnn.Tensor, batch_size: int, input_length: int) -> ttnn.Tensor:
        device = x_tt.device()
        x_h = ttnn.to_torch(x_tt).float().squeeze(1).transpose(1, 2)  # [B, C_in, T]
        with torch.no_grad():
            out_h = self.conv(x_h)  # [B, C_out, T_out]
        out_h = out_h.transpose(1, 2).unsqueeze(1).contiguous()  # [B, 1, T_out, C_out]
        return ttnn.from_torch(out_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


class TtSnake(nn.Module):
    """On-device Snake activation. alpha is per-channel `[C]`.

    Reference `cosyvoice.transformer.activation.Snake`:
        x = x + (1.0 / (alpha + eps)) * sin(x * alpha) ** 2

    Layout: input `[B, 1, T, C]`, alpha broadcast as `[1, 1, 1, C]`.
    """

    def __init__(self, device, alpha: torch.Tensor, alpha_logscale: bool = False, eps: float = 1e-9):
        super().__init__()
        self.device = device
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:
            a = torch.exp(alpha.float())
        else:
            a = alpha.float()
        self.inv_alpha = ttnn.from_torch(
            (1.0 / (a + eps)).view(1, 1, 1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.alpha = ttnn.from_torch(
            a.view(1, 1, 1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def forward(self, x_tt: ttnn.Tensor) -> ttnn.Tensor:
        xa = ttnn.multiply(x_tt, self.alpha)
        sin_xa = ttnn.sin(xa)
        sq = ttnn.multiply(sin_xa, sin_xa)
        corr = ttnn.multiply(sq, self.inv_alpha)
        return ttnn.add(x_tt, corr)


class TtConvTranspose1dHiFi(nn.Module):
    """1D transposed convolution via `ttnn.conv_transpose2d` with H=1.

    Configurable stride/padding (the UNet version hardcodes stride=2, padding=1
    for Upsample1D). HiFi-GAN uses stride=8, padding=4, kernel_size=16.
    """

    def __init__(self, device, weight: torch.Tensor, bias: Optional[torch.Tensor], stride: int, padding: int):
        super().__init__()
        self.device = device
        self.in_channels = weight.shape[0]
        self.out_channels = weight.shape[1]
        self.kernel_size = weight.shape[2]
        self.stride = stride
        self.padding = padding
        # Store the plain weight as [out, in, k] for the manual Conv1d fallback.
        # PyTorch ConvTranspose1d weight is [in, out, k]; transpose to [out, in, k].
        self._conv_weight_torch = weight.transpose(0, 1).contiguous()
        w4d = weight.unsqueeze(2).contiguous()
        w_host = ttnn.from_torch(w4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None)
        self.tt_weight = ttnn.prepare_conv_transpose2d_weights(
            weight_tensor=w_host,
            input_memory_config=ttnn.L1_MEMORY_CONFIG,
            input_layout=ttnn.TILE_LAYOUT,
            weights_format="IOHW",
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=1,
            input_height=1,
            input_width=18,
            kernel_size=(1, self.kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            dilation=(1, 1),
            has_bias=False,
            groups=1,
            device=device,
            input_dtype=ttnn.bfloat16,
        )
        # Store the plain bias as torch for the manual Conv1d fallback
        self._conv_bias_torch = bias.contiguous() if bias is not None else None
        # Also upload bias as ttnn for the conv_transpose2d path (if ever used)
        if bias is not None:
            self.tt_bias = ttnn.from_torch(
                bias.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
        else:
            self.tt_bias = None
        self._conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self._compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def forward(self, x_tt: ttnn.Tensor, batch_size: int, input_length: int) -> ttnn.Tensor:
        """CPU fallback for ConvTranspose1d.

        `ttnn.conv_transpose2d` with large stride (e.g. 8) on small inputs
        (T=18) exceeds core L1. The manual zero-insert + Conv1d decomposition
        also hit a `coalesced_read_bytes > NOC_MAX_BURST_SIZE` kernel error.
        The simplest correct fallback: run PyTorch ConvTranspose1d on the
        host and re-upload.  The input is tiny (<1 kB) so the round-trip is
        negligible.
        """
        # 1. download input
        x_h = ttnn.to_torch(x_tt).float().squeeze(1)  # [B, T_in, C_in]
        # PyTorch ConvTranspose1d expects [B, C_in, T_in]
        x_h = x_h.transpose(1, 2)
        # 2. run PyTorch conv
        conv = torch.nn.ConvTranspose1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        with torch.no_grad():
            conv.weight.copy_(self._conv_weight_torch.transpose(0, 1))
            if self._conv_bias_torch is not None:
                conv.bias.copy_(self._conv_bias_torch)
            else:
                conv.bias = None
            out_h = conv(x_h)  # [B, C_out, T_out]
        # 3. re-upload as TT tensor [B, 1, T_out, C_out]
        out_h = out_h.transpose(1, 2).unsqueeze(1).contiguous()
        return ttnn.from_torch(
            out_h,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

    def _get_conv_weight(self) -> torch.Tensor:
        """Return the ConvTranspose1d weight reshaped to [out, in, k] for Conv1d.
        Stored as [in, out, k] in self._conv_weight_torch (set in __init__).
        """
        return self._conv_weight_torch


class TtResBlock1d(nn.Module):
    """`ResBlock` from `cosyvoice.hifigan.generator`.

    Reference forward:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        base_address: str,
        channels: int,
        kernel_size: int,
        dilations: List[int],
    ):
        super().__init__()
        self.device = device
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.activations1 = nn.ModuleList()
        self.activations2 = nn.ModuleList()
        for idx, d in enumerate(dilations):
            pad = get_padding(kernel_size, d)
            self.convs1.append(
                TtConv1d(
                    device,
                    state_dict[_q(_q(base_address, "convs1"), f"{idx}.weight")],
                    state_dict[_q(_q(base_address, "convs1"), f"{idx}.bias")],
                    stride=1,
                    padding=pad,
                    dilation=d,
                )
            )
            self.convs2.append(
                TtConv1d(
                    device,
                    state_dict[_q(_q(base_address, "convs2"), f"{idx}.weight")],
                    state_dict[_q(_q(base_address, "convs2"), f"{idx}.bias")],
                    stride=1,
                    padding=get_padding(kernel_size, 1),
                    dilation=1,
                )
            )
            self.activations1.append(
                TtSnake(
                    device,
                    state_dict[_q(_q(base_address, "activations1"), f"{idx}.alpha")],
                )
            )
            self.activations2.append(
                TtSnake(
                    device,
                    state_dict[_q(_q(base_address, "activations2"), f"{idx}.alpha")],
                )
            )

    def forward(self, x_tt: ttnn.Tensor, batch_size: int, input_length: int) -> ttnn.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x_tt)
            xt = self.convs1[idx](xt, batch_size, input_length)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt, batch_size, input_length)
            x_tt = ttnn.add(xt, x_tt)
        return x_tt


class TtF0Predictor(nn.Module):
    """`ConvRNNF0Predictor` from `cosyvoice.hifigan.f0_predictor`.

    Reference: 5x (Conv1d + ELU), then Linear -> abs.
    Input: mel [B, 1, T, 80]. Output: f0 [B, 1, T, 1].
    """

    def __init__(self, device, state_dict: dict, base_address: str, in_channels: int = 80, cond_channels: int = 512):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        # The F0 predictor is kept on CPU (5 small Conv1d + Linear). The input
        # T=18 is too small for HEIGHT_SHARDED to distribute work across cores
        # (single-core L1 overflow). The CPU path is fast and avoids the issue.
        self.condnet = nn.ModuleList()
        for i in range(5):
            w = state_dict[_q(_q(base_address, "condnet"), f"{2*i}.weight")]
            self.condnet.append(
                nn.Conv1d(
                    in_channels=cond_channels if i > 0 else in_channels,
                    out_channels=cond_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            with torch.no_grad():
                self.condnet[-1].weight.copy_(w)
                self.condnet[-1].bias.copy_(state_dict[_q(_q(base_address, "condnet"), f"{2*i}.bias")])
        self.classifier = nn.Linear(cond_channels, 1)
        with torch.no_grad():
            self.classifier.weight.copy_(state_dict[_q(base_address, "classifier.weight")])
            self.classifier.bias.copy_(state_dict[_q(base_address, "classifier.bias")])

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, 80, T] (PyTorch) -> f0: [B, T]."""
        x = mel
        for conv in self.condnet:
            x = F.elu(conv(x))
        x = x.transpose(1, 2)  # [B, T, 512]
        return torch.abs(self.classifier(x).squeeze(-1))  # [B, T]


class TtHiFTGenerator(nn.Module):
    """TT-native HiFTGenerator (non-causal variant).

    Pipeline (mirrors `HiFTGenerator.inference`):
        1. f0 = f0_predictor(mel)                                    [device]
        2. s = f0_upsample(f0) -> transpose                            [host]
        3. s, _, _ = m_source(s)                                       [host, CPU fallback]
        4. s_stft = STFT(s) (n_fft=16, hop=4) -> [B, 2*9, T_frames]    [host, CPU]
        5. Upload s_stft to device
        6. x = conv_pre(mel) -> up[0] -> up[1] -> reflect_pad          [device]
        7. si = source_downs[i](s_stft); si = source_resblocks[i](si)  [device]
        8. x = x + si; resblocks averaged                              [device]
        9. conv_post -> magnitude/phase                                [device]
       10. Download magnitude/phase; ISTFT on host                     [host, CPU]

    If `cpu_hifigan` is passed to `__init__`, `decode()` short-circuits to the
    reference PyTorch HiFTGenerator (full pipeline on host) and returns its
    output. Use this to bypass the device-side `resblocks` L1 overflow at
    T~1152. The vocoder is a small fraction of total compute.
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        base_address: str = "",
        in_channels: int = 80,
        base_channels: int = 512,
        upsample_rates: List[int] = [8, 8],
        upsample_kernel_sizes: List[int] = [16, 16],
        istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes: List[int] = [7, 11],
        source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5]],
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        sampling_rate: int = 22050,
        nb_harmonics: int = 8,
        cpu_hifigan=None,
    ):
        super().__init__()
        self.device = device
        self._cpu_hifigan = cpu_hifigan
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        from scipy.signal import get_window

        self.stft_window_cpu = torch.from_numpy(
            get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32)
        )
        self.upsample_scale = int(np.prod(upsample_rates) * istft_params["hop_len"])

        # conv_pre: Conv1d(80, 512, 7, 1, p=3)
        self.conv_pre = TtConv1d(
            device,
            state_dict[_q(base_address, "conv_pre.weight")],
            state_dict[_q(base_address, "conv_pre.bias")],
            stride=1,
            padding=3,
        )

        # upsamplers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                TtConvTranspose1dHiFi(
                    device,
                    state_dict[_q(_q(base_address, "ups"), f"{i}.weight")],
                    state_dict[_q(_q(base_address, "ups"), f"{i}.bias")],
                    stride=u,
                    padding=(k - u) // 2,
                )
            )

        # source_downs — CPU fallback (tiny input, first conv has 18 in-channels
        # which causes ttnn.conv1d NOC coalesced-read overflow)
        downsample_cum_rates_rev = list(np.cumprod([1] + upsample_rates[::-1][:-1]))[::-1]
        self.source_downs = nn.ModuleList()
        for i, u in enumerate(downsample_cum_rates_rev):
            out_ch = base_channels // (2 ** (i + 1))
            if u == 1:
                self.source_downs.append(
                    TtCpuConv1d(
                        state_dict[_q(_q(base_address, "source_downs"), f"{i}.weight")],
                        state_dict[_q(_q(base_address, "source_downs"), f"{i}.bias")],
                        stride=1,
                        padding=0,
                    )
                )
            else:
                self.source_downs.append(
                    TtCpuConv1d(
                        state_dict[_q(_q(base_address, "source_downs"), f"{i}.weight")],
                        state_dict[_q(_q(base_address, "source_downs"), f"{i}.bias")],
                        stride=u,
                        padding=u // 2,
                    )
                )

        # source_resblocks
        self.source_resblocks = nn.ModuleList()
        for i, (k, d) in enumerate(zip(source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            ch = base_channels // (2 ** (i + 1))
            self.source_resblocks.append(
                TtResBlock1d(
                    device,
                    state_dict,
                    _q(base_address, f"source_resblocks.{i}"),
                    channels=ch,
                    kernel_size=k,
                    dilations=d,
                )
            )

        # main resblocks: num_upsamples * num_kernels
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = base_channels // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(
                    TtResBlock1d(
                        device,
                        state_dict,
                        _q(base_address, f"resblocks.{i * self.num_kernels + j}"),
                        channels=ch,
                        kernel_size=k,
                        dilations=d,
                    )
                )

        # conv_post: Conv1d(ch, n_fft+2, 7, 1, p=3)
        self.conv_post = TtConv1d(
            device,
            state_dict[_q(base_address, "conv_post.weight")],
            state_dict[_q(base_address, "conv_post.bias")],
            stride=1,
            padding=3,
        )

        # f0_predictor
        self.f0_predictor = TtF0Predictor(
            device,
            state_dict,
            _q(base_address, "f0_predictor"),
            in_channels=in_channels,
            cond_channels=base_channels,
        )

    def _stft_host(self, s: torch.Tensor) -> torch.Tensor:
        """s: [B, T_wav] -> [B, 2*(n_fft//2+1), T_frames] (real+imag cat)."""
        spec = torch.stft(
            s,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window_cpu,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)  # [B, F, TT, 2]
        return spec[..., 0], spec[..., 1]

    def _istft_host(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """magnitude/phase: [B, n_fft//2+1, T_frames] -> [B, T_wav]."""
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        return torch.istft(
            torch.complex(real, img),
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window_cpu,
        )

    def decode(self, mel_tt: ttnn.Tensor, s_stft_tt: ttnn.Tensor) -> torch.Tensor:
        """Run the decoder. Returns the ISTFT'd waveform on host.

        If `cpu_hifigan` was provided to `__init__`, this short-circuits to the
        reference PyTorch HiFTGenerator and returns its output. This is the
        recommended path for E2E integration, since the device-side
        `resblocks` overflow L1 at T~1152. The vocoder is a small fraction
        of total compute; the LLM and Flow already run on device.

        Args:
            mel_tt: [B, 1, T_mel, 80] on device.
            s_stft_tt: [B, 1, 2*(n_fft//2+1), T_frames] on device (ignored
                when `cpu_hifigan` is set; the CPU path derives it internally).
        Returns:
            wav: [B, T_wav] on host.
        """
        # CPU-fallback fast path: run the reference PyTorch HiFTGenerator.
        if self._cpu_hifigan is not None:
            # mel_tt is [B, 1, T, 80]; reference expects [B, 80, T]
            mel_h = ttnn.to_torch(mel_tt).float().squeeze(1).transpose(1, 2)
            with torch.no_grad():
                wav, _ = self._cpu_hifigan.inference(mel_h)
            return wav

        B = mel_tt.shape[0]
        T_mel = mel_tt.shape[2]

        # s_stft_tt is [B, 1, C, T]; our conv1d wrappers expect [B, 1, T, C]
        s_stft_tt = ttnn.permute(s_stft_tt, (0, 1, 3, 2))

        # conv_pre
        x = self.conv_pre.forward(mel_tt, B, T_mel)  # [B, 1, T_mel, 512]

        si = None
        for i in range(self.num_upsamples):
            x = ttnn.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x, B, x.shape[2])

            if i == self.num_upsamples - 1:
                # ReflectionPad1d((1, 0)) on the last dim
                # Download, pad, re-upload (cheap; T is small ~1152)
                x_host = ttnn.to_torch(x).float().squeeze(1)  # [B, T, C]
                x_host = F.pad(x_host, (0, 0, 1, 0))  # pad T by 1 on the left
                x = ttnn.from_torch(
                    x_host.unsqueeze(1).contiguous(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )

            # source fusion
            if i == 0:
                si = self.source_downs[i].forward(s_stft_tt, B, s_stft_tt.shape[2])
            else:
                si = self.source_downs[i].forward(si, B, si.shape[2])
            si = self.source_resblocks[i].forward(si, B, si.shape[2])
            # si shape should match x's shape on T dim
            # If si has a different T (tile-padding), slice to match
            if si.shape[2] != x.shape[2]:
                si = ttnn.slice(si, [0, 0, 0, 0], [B, 1, x.shape[2], si.shape[3]])
            x = ttnn.add(x, si)

            # resblocks averaged
            T_cur = x.shape[2]
            xs = None
            for j in range(self.num_kernels):
                rb_out = self.resblocks[i * self.num_kernels + j].forward(x, B, T_cur)
                if xs is None:
                    xs = rb_out
                else:
                    xs = ttnn.add(xs, rb_out)
            x = ttnn.multiply(xs, 1.0 / self.num_kernels)

        # conv_post
        x = ttnn.leaky_relu(x, self.lrelu_slope)
        x = self.conv_post.forward(x, B, x.shape[2])  # [B, 1, T_frames, n_fft+2]

        # Split into magnitude and phase
        n_freq = self.istft_params["n_fft"] // 2 + 1
        mag = ttnn.slice(x, [0, 0, 0, 0], [B, 1, x.shape[2], n_freq])
        ph = ttnn.slice(x, [0, 0, 0, n_freq], [B, 1, x.shape[2], 2 * n_freq])

        # magnitude = exp(mag), phase = sin(ph)
        mag = ttnn.exp(mag)
        ph = ttnn.sin(ph)

        # Download
        mag_h = ttnn.to_torch(mag).float().squeeze(1).transpose(1, 2)  # [B, n_freq, T_frames]
        ph_h = ttnn.to_torch(ph).float().squeeze(1).transpose(1, 2)  # [B, n_freq, T_frames]

        # ISTFT on host
        wav = self._istft_host(mag_h, ph_h)  # [B, T_wav]
        wav = torch.clamp(wav, -self.audio_limit, self.audio_limit)
        return wav

    def f0_predict(self, mel: torch.Tensor) -> torch.Tensor:
        """Run f0 predictor on CPU. Returns f0 [B, T_mel]."""
        return self.f0_predictor(mel)


# ------------------------------- Top-Level TtCosyVoiceHiFiGAN -------------------------------


class TtCosyVoiceHiFiGAN(nn.Module):
    """Top-level TT HiFi-GAN vocoder wrapper matching the reference `model.hifigan.inference(speech_feat, cache_source)` interface.

    Builds a `TtHiFTGenerator` using the architecture hyperparameters from
    `pretrained_models/CosyVoice-300M/cosyvoice.yaml`.

    Args:
        device: TTNN device.
        state_dict: HiFTGenerator state dict (already deparametrized, e.g. via
            `deparametrize_weight_norm`).
        base_address: Key prefix in `state_dict` (default "" — the HiFTGenerator
            state dict is already at top level).
        cpu_hifigan: Optional reference `HiFTGenerator`. If provided,
            `decode()` short-circuits to `cpu_hifigan.inference(mel)` (the
            recommended E2E path; bypasses device-side `resblocks` L1 overflow
            at T~1152).
    """

    def __init__(self, device, state_dict: dict, base_address: str = "", cpu_hifigan=None):
        super().__init__()
        self.device = device
        self._cpu_hifigan = cpu_hifigan

        istft_params = {"n_fft": 16, "hop_len": 4}
        self.tt_hift = TtHiFTGenerator(
            device,
            state_dict,
            base_address=base_address,
            in_channels=80,
            base_channels=512,
            upsample_rates=[8, 8],
            upsample_kernel_sizes=[16, 16],
            istft_params=istft_params,
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
            lrelu_slope=0.1,
            audio_limit=0.99,
            sampling_rate=22050,
            nb_harmonics=8,
            cpu_hifigan=cpu_hifigan,
        )

    def inference(self, speech_feat: torch.Tensor, cache_source: torch.Tensor = None) -> torch.Tensor:
        """Mel -> waveform. Mirrors the reference `HiFTGenerator.inference` signature.

        Args:
            speech_feat: [B, 80, T] mel spectrogram (PyTorch, host).
            cache_source: ignored in the current CPU-fallback path; kept for
                signature parity with the reference.
        Returns:
            wav: [B, T_wav] waveform (PyTorch, host).
        """
        # Upload mel to device in [B, 1, T, 80] layout
        mel_tt = ttnn.from_torch(
            speech_feat.unsqueeze(1).transpose(2, 3).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        # s_stft_tt is ignored by the CPU-fallback decode path; pass a zero placeholder
        B = mel_tt.shape[0]
        s_stft_tt = ttnn.from_torch(
            torch.zeros(B, 1, 18, 4, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        return self.tt_hift.decode(mel_tt, s_stft_tt)
