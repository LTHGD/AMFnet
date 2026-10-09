"""MFBF：多级特征双向融合模块（论文 Eq.13 - Eq.16，Fig.5）。

作用：在解码阶段整合浅层细节、中层结构、深层语义三级特征，实现
"自下而上细节传递 + 自上而下语义引导 + 中层校准"的双向交互，
提升不完整缺陷的定位精度与分类置信度。

三个注意力组件（均输出 [0,1] 权重图，用于 Eq.16 的元素级相乘）：
  - PCA  像素级通道注意力（Eq.13）：为浅层特征每个空间位置计算独立
         通道权重，细粒度传递微小缺陷纹理（源自 SENet 的逐像素变体）。
  - SCA  空间-通道注意力（Eq.14）：并行生成空间注意力 M_s（GAP+GMP
         通道池化）与通道注意力 M_c，捕获不连续缺陷轮廓（源自 CBAM）。
  - GCA  全局通道注意力（Eq.15）：对深层聚合语义做全局通道筛选，
         自上而下传递高层语义（源自 SENet）。

最终融合（Eq.16）：
  F_MBFB = (F^L ⊗ f_SCA(F^M') ⊗ f_GCA(F^H'))
         + (F^M' ⊗ f_PCA(F^L) ⊗ f_GCA(F^H'))
         + (F^H' ⊗ f_PCA(F^L) ⊗ f_SCA(F^M'))
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


class PCA(nn.Module):
    """像素级通道注意力（Eq.13）。

    f_PCA(F) = delta( BN( W_p2 * sigma( BN( W_p1 * F ) ) ) )，返回权重图。
    """

    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or max(channels // 4, 8)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        w = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return torch.sigmoid(w)                                # 逐像素通道权重 [B, C, H, W]


class SCA(nn.Module):
    """空间-通道注意力（Eq.14）。

    f_SCA(F) = M_s ⊗ (M_c ⊙ F)；此处返回权重图 M_s ⊗ M_c（形状 [B,C,H,W]），
    供 Eq.16 直接与特征元素级相乘。
    """

    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or max(channels // 4, 8)
        # M_s：空间注意力（通道维 max/avg 池化拼接 -> 3x3 卷积）
        self.conv_s = nn.Conv2d(2, 1, 3, 1, 1, bias=False)
        self.bn_s = nn.BatchNorm2d(1)
        # M_c：通道注意力（GAP/GMP 描述子 -> 1x1 卷积，1x1 空间用 GroupNorm）
        self.conv_c1 = nn.Conv2d(channels * 2, hidden, 1, bias=False)
        self.bn_c1 = nn.GroupNorm(1, hidden)
        self.conv_c2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn_c2 = nn.GroupNorm(1, channels)

    def forward(self, x):
        # 空间分支
        x_max = x.max(dim=1, keepdim=True)[0]
        x_avg = x.mean(dim=1, keepdim=True)
        m_s = torch.sigmoid(self.bn_s(self.conv_s(torch.cat([x_max, x_avg], dim=1))))
        # 通道分支
        f_avg = F.adaptive_avg_pool2d(x, 1)
        f_max = F.adaptive_max_pool2d(x, 1)
        f_c = torch.cat([f_avg, f_max], dim=1)                 # F_concat (Eq.14)
        m_c = torch.sigmoid(self.bn_c2(
            self.conv_c2(F.relu(self.bn_c1(self.conv_c1(f_c))))))
        return m_s * m_c                                       # 权重图 [B, C, H, W]


class GCA(nn.Module):
    """全局通道注意力（Eq.15）。

    f_GCA(F) = M_g ⊙ F，M_g = delta( BN(Conv1x1( sigma(BN(Conv1x1(GAP(F)))) ) ) )
    """

    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or max(channels // 4, 8)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.GroupNorm(1, hidden)
        self.conv2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn2 = nn.GroupNorm(1, channels)

    def forward(self, x):
        f_avg = F.adaptive_avg_pool2d(x, 1)                    # GAP(F)
        m_g = torch.sigmoid(self.bn2(
            self.conv2(F.relu(self.bn1(self.conv1(f_avg))))))
        return m_g                                             # [B, C, 1, 1]


class MFBF(nn.Module):
    """MFBF 模块整体（Eq.16）。输入为 TFIC 对齐到中间尺度的三级特征。"""

    def __init__(self, channels, hidden=64):
        super().__init__()
        self.channels = channels
        self.pca = PCA(channels, hidden)
        self.sca = SCA(channels, hidden)
        self.gca = GCA(channels, hidden)
        self.out = ConvBlock(channels, channels, stride=1, kernel_size=1)

    def forward(self, f_l, f_m, f_h):
        """f_l / f_m / f_h: [B, C, H/2, W/2]（TFIC 输出，尺度已对齐）。"""
        w_pca = self.pca(f_l)                                  # f_PCA(F^L)
        w_sca = self.sca(f_m)                                  # f_SCA(F^M')
        w_gca = self.gca(f_h)                                  # f_GCA(F^H')
        # ---- Eq.16 三方交叉融合 ----
        out = (f_l * w_sca * w_gca) + (f_m * w_pca * w_gca) + (f_h * w_pca * w_sca)
        return self.out(out)

