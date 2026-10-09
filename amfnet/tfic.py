
"""TFIC：Transformer 特征信息补偿模块（论文 Eq.9 - Eq.12）。

目标：利用浅层细节与深层语义双向补偿中层特征，恢复不完整缺陷
（边界截断缺陷 / 低对比度内部缺陷）在下采样中丢失的细节信息。

核心组件：
  - GCB  全局上下文块（Eq.9）：源自 GCNet，含三点针对性改进
       ① 引入 GAT-FE 传来的缺陷置信度权重，动态放大缺陷区域响应；
       ② 双分支注意力分别补偿"边界截断"与"内部低对比度"两类缺陷；
       ③ 内嵌卷积上/下采样分支实现多尺度信息匹配。
  - CDB  卷积下采样块（Eq.11）：f_CDB = sigma(BN(Conv_3x3(.)))

双向补偿结构（Eq.12），三个分支输出均对齐到中间尺度 H/2 x W/2：
    F_TIC^L = Down(GCB(F_L))                      浅层细节 -> 中间尺度
    F_TIC^M = CDB(GCB(F_M)) + F_M                 中层自增强（保持尺度）
    F_TIC^H = Up(GCB(CDB(F_H))) + Up(F_H)         深层语义 -> 中间尺度
    F_TIC   = F_TIC^L + F_TIC^M + F_TIC^H
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv + BN + ReLU 基本单元（局部定义）。"""

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力，支持局部窗口化（attn_window > 0 且小于特征尺寸时）。"""

    def __init__(self, channels, heads=8, window=0, dropout=0.1):
        super().__init__()
        assert channels % heads == 0, "channels 必须能被 heads 整除"
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.window = window
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

    def _attn(self, tokens):
        """tokens: [B, T, C] -> [B, T, C]"""
        b, t, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)         # 各 [B, H, T, D]
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                            # [B, H, T, D]
        out = out.permute(0, 2, 1, 3).reshape(b, t, self.channels)
        return self.proj(out)

    def forward(self, x):
        """x: [B, C, H, W] -> [B, C, H, W]"""
        b, c, h, w = x.shape
        ws = self.window
        if ws > 0 and (h > ws or w > ws):
            # ---- 局部窗口注意力：切分为 ws x ws 窗口 ----
            ph = (ws - h % ws) % ws
            pw = (ws - w % ws) % ws
            xp = F.pad(x, (0, pw, 0, ph))
            _, _, hh, ww = xp.shape
            tokens = (xp.reshape(b, c, hh // ws, ws, ww // ws, ws)
                        .permute(0, 2, 4, 3, 5, 1)
                        .reshape(b, -1, ws * ws, c))           # [B, num_win, K, C]
            out = self._attn(tokens.reshape(-1, ws * ws, c))
            out = out.reshape(b, hh // ws, ww // ws, ws, ws, c)
            out = out.permute(0, 5, 1, 3, 2, 4).reshape(b, c, hh, ww)
            return out[:, :, :h, :w]
        # ---- 全局注意力 ----
        tokens = x.flatten(2).transpose(1, 2)                  # [B, H*W, C]
        out = self._attn(tokens)
        return out.transpose(1, 2).reshape(b, c, h, w)


class GlobalContextBlock(nn.Module):
    """全局上下文块 GCB（Eq.9）。

    z_i = x_i + W_v2 ReLU(LN(W_v1 * z_global))
    结构：Conv1x1 -> 多头自注意力 -> Conv1x1 -> LayerNorm+ReLU -> Conv1x1（残差）。
    双分支：self.attn_a / self.attn_b 分别对应边界截断缺陷与内部低对比度缺陷
    的差异化特征恢复，输出经可学习标量 self.mix 融合。
    """

    def __init__(self, channels, heads=8, window=0, dropout=0.1, dual_branch=True):
        super().__init__()
        self.channels = channels
        self.dual_branch = dual_branch
        self.conv_in = nn.Conv2d(channels, channels, 1)
        self.attn_a = MultiHeadSelfAttention(channels, heads, window, dropout)
        if dual_branch:
            self.attn_b = MultiHeadSelfAttention(channels, heads, window, dropout)
        # 双分支融合系数（可学习，初始 0.5）
        self.mix = nn.Parameter(torch.tensor(0.5))
        # 缺陷置信度加权强度（可学习，初始 1.0）
        self.conf_scale = nn.Parameter(torch.tensor(1.0))
        self.conv_v1 = nn.Conv2d(channels, channels, 1)
        # LayerNorm：对每个空间位置在通道维归一化（GroupNorm(1, C) 等价近似）
        self.ln = nn.GroupNorm(1, channels)
        self.conv_v2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x, conf=None):
        z = self.conv_in(x)
        if conf is not None:
            conf = F.interpolate(conf, size=z.shape[-2:], mode="bilinear",
                                 align_corners=False)
            z = z * (1 + self.conf_scale * conf)               # 缺陷置信度加权（改进点①）
        za = self.attn_a(z)
        if self.dual_branch:
            zb = self.attn_b(z)                                # 双分支（改进点②）
            z = za + self.mix * zb
        else:
            z = za
        z = self.conv_v1(z)                                    # W_v1
        z = self.ln(z)
        z = torch.relu(z)
        z = self.conv_v2(z)                                    # W_v2
        return x + z                                           # Eq.9 残差


class ConvDownsampleBlock(nn.Module):
    """卷积下采样块 CDB（Eq.11）：f_CDB(.) = sigma(BN(Conv_3x3(.)))。"""

    def __init__(self, channels, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TFIC(nn.Module):
    """TFIC 模块整体（Eq.9 - Eq.12）。"""

    def __init__(self, low_channels=64, mid_channels=128, high_channels=256,
                 out_channels=256, attn_heads=8, attn_window=32,
                 dropout=0.1, dual_branch=True):
        super().__init__()
        self.out_channels = out_channels
        # 通道对齐投影
        self.proj_l = ConvBlock(low_channels, out_channels, stride=1, kernel_size=1)
        self.proj_m = ConvBlock(mid_channels, out_channels, stride=1, kernel_size=1)
        self.proj_h = ConvBlock(high_channels, out_channels, stride=1, kernel_size=1)
        # 三个尺度的 GCB
        self.gcb_l = GlobalContextBlock(out_channels, attn_heads, attn_window,
                                        dropout, dual_branch)
        self.gcb_m = GlobalContextBlock(out_channels, attn_heads, attn_window,
                                        dropout, dual_branch)
        self.gcb_h = GlobalContextBlock(out_channels, attn_heads, attn_window,
                                        dropout, dual_branch)
        # 双向补偿分支（Eq.12）
        self.cdb_l = ConvDownsampleBlock(out_channels, stride=2)   # 浅层 -> 1/2 尺度
        self.cdb_m = ConvDownsampleBlock(out_channels, stride=1)   # 中层保持尺度
        self.cdb_h = ConvDownsampleBlock(out_channels, stride=1)   # 深层瓶颈变换

    def forward(self, f_l, f_m, f_h, conf=None):
        """输入 GAT-FE 输出（均已在 GAT-FE 内投影对齐到 out_channels）。

        返回 (F_TIC^L, F_TIC^M, F_TIC^H, F_TIC)，三个分支与总和均为中间尺度。
        """
        f_l = self.proj_l(f_l)                                # [B, C, H, W]
        f_m = self.proj_m(f_m)                                # [B, C, H/2, W/2]
        f_h = self.proj_h(f_h)                                # [B, C, H/4, W/4]

        # 浅层分支：GCB -> CDB(2x 下采样) -> 中间尺度
        ft_l = self.cdb_l(self.gcb_l(f_l, conf))
        # 中层分支：GCB -> CDB(1x) + 残差（保持中间尺度）
        ft_m = self.cdb_m(self.gcb_m(f_m, conf)) + f_m
        # 深层分支：CDB(1x) -> GCB -> 2x 上采样 + 上采样残差 -> 中间尺度
        ft_h = F.interpolate(self.gcb_h(self.cdb_h(f_h), conf), scale_factor=2,
                             mode="bilinear", align_corners=False)
        ft_h = ft_h + F.interpolate(f_h, scale_factor=2, mode="bilinear",
                                    align_corners=False)

        f_tic = ft_l + ft_m + ft_h                            # Eq.12
        return ft_l, ft_m, ft_h, f_tic
