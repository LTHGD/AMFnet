
"""AMFNet 基础特征提取（论文 Eq.1）。

三级初始特征并行地从输入芯片图像 I_p 提取：
    F_L^0(X) = Conv_L(I_p) in R^{H x W x C_l}      低层细节特征
    F_M^0(X) = Conv_M(I_p) in R^{H/2 x W/2 x C_m}  中层结构特征
    F_H^0(X) = Conv_H(I_p) in R^{H/4 x W/4 x C_h}  高层语义特征

Conv_L / Conv_M / Conv_H 均为 Conv+BN+ReLU 堆叠分支，通过步长控制分辨率，
作为 GAT-FE 模块的图节点初始化输入。
"""

import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv + BN + ReLU 基本单元。"""

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class AMFBackbone(nn.Module):
    """按 Eq.1 从输入图像并行提取低 / 中 / 高三级初始特征。"""

    def __init__(self, in_channels=3, low_channels=64,
                 mid_channels=128, high_channels=256):
        super().__init__()
        # 低层分支：保持分辨率 H x W
        self.conv_l = nn.Sequential(
            ConvBlock(in_channels, low_channels // 2, stride=1),
            ConvBlock(low_channels // 2, low_channels, stride=1),
        )
        # 中层分支：下采样 2 倍 -> H/2 x W/2
        self.conv_m = nn.Sequential(
            ConvBlock(in_channels, mid_channels // 2, stride=2),
            ConvBlock(mid_channels // 2, mid_channels, stride=1),
        )
        # 高层分支：下采样 4 倍 -> H/4 x W/4
        self.conv_h = nn.Sequential(
            ConvBlock(in_channels, high_channels // 4, stride=2),
            ConvBlock(high_channels // 4, high_channels // 2, stride=2),
            ConvBlock(high_channels // 2, high_channels, stride=1),
        )

    def forward(self, x):
        f_l = self.conv_l(x)   # [B, C_l, H, W]
        f_m = self.conv_m(x)   # [B, C_m, H/2, W/2]
        f_h = self.conv_h(x)   # [B, C_h, H/4, W/4]
        return f_l, f_m, f_h
