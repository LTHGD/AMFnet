


import torch
import torch.nn as nn

from .backbone import AMFBackbone
from .gatfe import GATFE
from .tfic import TFIC
from .mfbf import MFBF
from .head import CenterNetHead


class AMFNet(nn.Module):
    """Information Compensation-based Adaptive Multi-scale Fusion Network。"""

    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.backbone = AMFBackbone(
            in_channels=m.in_channels,
            low_channels=m.low_channels,
            mid_channels=m.mid_channels,
            high_channels=m.high_channels,
        )
        self.gatfe = GATFE(
            low_channels=m.low_channels,
            mid_channels=m.mid_channels,
            high_channels=m.high_channels,
            tfic_channels=m.tfic_channels,
            low_patch=m.gatfe.low_patch,
            mid_patch=m.gatfe.mid_patch,
            knn_window=m.gatfe.knn_window,
            tau_L=m.gatfe.tau_L,
            tau_M=m.gatfe.tau_M,
            tau_H=m.gatfe.tau_H,
            hidden=m.gatfe.hidden,
        )
        self.tfic = TFIC(
            low_channels=m.tfic_channels,
            mid_channels=m.tfic_channels,
            high_channels=m.tfic_channels,
            out_channels=m.tfic_channels,
            attn_heads=m.tfic.attn_heads,
            attn_window=m.tfic.attn_window,
            dropout=m.tfic.dropout,
            dual_branch=m.tfic.dual_branch,
        )
        self.mfbf = MFBF(channels=m.tfic_channels, hidden=m.mfbf.hidden)
        self.head = CenterNetHead(
            in_channels=m.tfic_channels,
            num_classes=m.num_classes,
            hm_hidden=m.head.hm_hidden,
        )

    def forward(self, x):
        """x: [B, 3, H, W] -> pred dict{heatmap, offset, size}"""
        f_l0, f_m0, f_h0 = self.backbone(x)                    # Eq.1
        f_l1, f_m1, f_h1, _fused, conf = self.gatfe(f_l0, f_m0, f_h0)  # Eq.2-8
        ft_l, ft_m, ft_h, f_tic = self.tfic(f_l1, f_m1, f_h1, conf)    # Eq.9-12
        f_fuse = self.mfbf(ft_l, ft_m, ft_h)                   # Eq.13-16
        pred = self.head(f_fuse)                               # 分类 + 定位
        return pred

    def forward_features(self, x):
        """前向并返回中间特征（供分析/可视化/消融使用）。"""
        f_l0, f_m0, f_h0 = self.backbone(x)
        f_l1, f_m1, f_h1, fused, conf = self.gatfe(f_l0, f_m0, f_h0)
        ft_l, ft_m, ft_h, f_tic = self.tfic(f_l1, f_m1, f_h1, conf)
        f_fuse = self.mfbf(ft_l, ft_m, ft_h)
        pred = self.head(f_fuse)
        return {
            "pred": pred, "f_l0": f_l0, "f_m0": f_m0, "f_h0": f_h0,
            "f_l1": f_l1, "f_m1": f_m1, "f_h1": f_h1,
            "gatfe_fused": fused, "conf": conf,
            "f_tic_l": ft_l, "f_tic_m": ft_m, "f_tic_h": ft_h, "f_tic": f_tic,
            "f_mfbf": f_fuse,
        }
