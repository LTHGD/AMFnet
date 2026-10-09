"""GAT-FE：基于图注意力的特征提取模块（论文 Eq.2 - Eq.8）。

组成（对应论文 Fig.2）：
  - L-GNA   Low-level Graph Neighborhood Attention       Eq.2 / Eq.5
  - M-CGSA  Mid-level Context-aware Graph Separable Attention  Eq.3 / Eq.6
  - H-CGSA  High-level Global Context Semantic Attention Eq.4 / Eq.7
  - 三级特征对齐融合（三重注意力加权）                      Eq.8

实现说明：
  * 论文中图节点数 N_L = H/2 x W/2、N_M = H/4 x W/4、N_H = C_h。
    本实现低/中层特征经 AvgPool 得到节点网格，邻居取空间 K-NN 窗口
    （window x window），避免 N x N 全连接邻接矩阵的显存开销；
    高层以“通道”为图节点做全局上下文注意力（C_h x C_h，开销可接受）。
  * A_M 中的 IOU(i,j) 论文定义为检测头预测框的 IoU，本实现用节点
    区域的固定空间重叠近似；DSim(i,j) 用 MLP 投影特征的余弦相似度。
  * 模块末端按 Eq.8 对齐拼接并做三重注意力加权（仅供模块内部全局
    加权使用），三级独立张量保留并送入下一模块（TFIC）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv + BN + ReLU 基本单元（局部定义，避免循环导入）。"""

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def _neighbor_offsets(window, device):
    """窗口内相对坐标偏移 [K, 2]（行优先展开），用于空间距离 / IoU 先验。"""
    r = window // 2
    offsets = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    return torch.tensor(offsets, dtype=torch.float32, device=device)


def _unfold_neighbors(x, window):
    """把特征图 x:[B,C,H,W] 展开为每个节点的 window x window 邻居集合。

    返回 (neighbors:[B,N,K,C], mask:[B,N,K])，其中 N=H*W, K=window^2。
    边界采用 reflect 填充，保证每个节点都有完整邻居窗口。
    """
    b, c, h, w = x.shape
    k = window * window
    pad = window // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    xn = F.unfold(xp, kernel_size=window, stride=1)          # [B, C*K, N]
    xn = xn.reshape(b, c, k, h * w).permute(0, 3, 2, 1)      # [B, N, K, C]
    return xn


class LGNABlock(nn.Module):
    """低层图邻域注意力（Eq.2 / Eq.5）：增强弱边界缺陷的局部纹理。

    图节点:  v_Li = AvgPool2d(F_L^0(X_i), 2) * w_i
             w_i  = sigmoid(MLP_pixel(v_Li)) in [0,1]  像素块缺陷先验置信度
    邻接:    A_L(ij) = exp(-d_L(i,j)/(2 d_Lmax^2)) * min(w_i, w_j)
    注意力:  a_Lij = A_L(ij)*exp(-||v_Li-v_Lj||^2/(2 tau_L^2)) / sum_k ...
    聚合:    F_L^(1)(X) = F_L^0(X) + sigma( sum_j a_Lij * MLP(v_Lj) )
    """

    def __init__(self, channels, patch=2, tau=1.0, window=3, hidden=None):
        super().__init__()
        self.patch = patch
        self.tau = tau
        self.window = window
        hidden = hidden or max(channels // 2, 16)
        # w_i = sigma(MLP_pixel(v_Li))：两层 MLP（以 1x1 卷积等价实现）
        self.mlp_pixel = nn.Sequential(
            nn.Conv1d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, 1), nn.Sigmoid(),
        )
        # MLP(v_j)：邻居节点特征变换
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch
        # ---- 图节点构造 (Eq.2) ----
        v = F.avg_pool2d(x, kernel_size=p, stride=p)          # [B, C, Hn, Wn]
        bn, cn, hn, wn = v.shape
        vf = v.reshape(bn, cn, -1)                            # [B, C, N]
        w_i = self.mlp_pixel(vf)                              # [B, 1, N] 缺陷置信度
        v = vf * w_i                                          # v_Li = AvgPool(F)*w_i

        # ---- 空间 K-NN 邻居窗口 ----
        nbr_v = _unfold_neighbors(v.reshape(bn, cn, hn, wn), self.window)  # [B, N, K, C]
        nbr_w = _unfold_neighbors(w_i.reshape(bn, 1, hn, wn), self.window)  # [B, N, K, 1]

        # 空间距离先验：exp(-d_L^2 / (2 d_Lmax^2))            (Eq.2)
        offs = _neighbor_offsets(self.window, x.device)
        d = offs.norm(dim=1)                                  # [K]
        a_spatial = torch.exp(-(d ** 2) / (2 * d.max() ** 2 + 1e-6))

        wi = w_i.reshape(bn, -1, 1, 1)                        # [B, N, 1, 1]
        # min(w_i, w_j)：节点自身置信度与邻居置信度逐元素取小
        min_w = torch.minimum(wi, nbr_w).squeeze(-1)          # [B, N, K]
        adj = a_spatial.view(1, 1, -1) * min_w                # A_L(ij) (Eq.2)

        # 特征距离高斯：exp(-||v_i-v_j||^2 / (2 tau_L^2))     (Eq.5)
        vi = vf.reshape(bn, cn, -1, 1)                        # [B, C, N, 1]
        nbr_t = nbr_v.permute(0, 3, 1, 2)                     # [B, C, N, K]
        diff = vi - nbr_t
        feat_sim = torch.exp(-(diff ** 2).sum(dim=1) / (2 * self.tau ** 2 + 1e-6))  # [B, N, K]
        a_ij = adj * feat_sim
        a_ij = a_ij / (a_ij.sum(dim=-1, keepdim=True) + 1e-6)  # 归一化 (Eq.5)

        # ---- 邻居特征变换与聚合 ----
        k = self.window * self.window
        mlp_in = nbr_v.reshape(bn, -1, cn).transpose(1, 2)    # [B, C, N*K]
        mlp_out = self.mlp(mlp_in).transpose(1, 2).reshape(bn, hn * wn, k, cn)
        agg = (a_ij.unsqueeze(-1) * mlp_out).sum(dim=2)       # [B, N, C]
        agg = agg.transpose(1, 2).reshape(bn, cn, hn, wn)
        agg = F.interpolate(agg, size=(h, w), mode="nearest")  # 恢复到 F_L^0 分辨率
        return x + torch.relu(agg)                            # F_L^(1) (Eq.5)


class MCGSABlock(nn.Module):
    """中层上下文感知图可分离注意力（Eq.3 / Eq.6）：描述不完整缺陷形态。

    图节点:  v_Mi = Conv(F_M^0(X_i)) * Softmax(DS(1..i))
             DS(i) = MLP_mid(v_Mi) 为候选缺陷区域缺陷概率得分
    邻接:    A_M(ij) = exp(-||v_Mi-v_Mj||^2/(2 tau_M^2)) * (IOU(i,j)+DSim(i,j))/2
    注意力:  h_Mi = MLP( concat(v_Mi, GMP({v_Mj}_{j in N_M(i)})) )
             A_Ms = Softmax(h_Mi h_Mj^T/sqrt(d) ⊙ A_M(ij))   空间可分离注意力
             A_Mc = Softmax(h_Mi^T h_Mj/sqrt(d))             通道注意力
    聚合:    F_M^(1)(X) = F_M^0(X) + sigma(A_Ms ⊙ A_Mc ⊙ h_Mi)
    """

    def __init__(self, channels, patch=2, tau=1.0, window=3, hidden=None):
        super().__init__()
        self.patch = patch
        self.tau = tau
        self.window = window
        hidden = hidden or max(channels // 2, 16)
        self.conv_node = ConvBlock(channels, channels, stride=1, kernel_size=3)
        self.mlp_mid = nn.Sequential(                         # DS(i)
            nn.Conv1d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, 1),
        )
        self.mlp_sem = nn.Sequential(                         # DSim 投影
            nn.Conv1d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
        )
        self.mlp_h = nn.Sequential(                           # h_Mi
            nn.Conv1d(channels * 2, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
        )

    def forward(self, x):
        b, c, H, W = x.shape                                  # F_M^0: [B, C, H/2, W/2]
        p = self.patch
        v = self.conv_node(x)
        v = F.avg_pool2d(v, kernel_size=p, stride=p)          # 节点网格 [B, C, H/4, W/4]
        bn, cn, hn, wn = v.shape
        k = self.window * self.window
        vf = v.reshape(bn, cn, -1)                            # [B, C, N]

        # DS(i) = MLP_mid(v_Mi)，Softmax 归一化后加权节点 (Eq.3)
        ds = self.mlp_mid(vf)                                 # [B, 1, N]
        v = vf * torch.softmax(ds, dim=2)

        # 邻居窗口与 GMP 上下文
        nbr_v = _unfold_neighbors(v.reshape(bn, cn, hn, wn), self.window)  # [B, N, K, C]
        gmp = nbr_v.max(dim=2).values.transpose(1, 2)         # [B, C, N]（节点自身邻域 GMP）
        h = self.mlp_h(torch.cat([vf, gmp], dim=1))           # h_Mi: [B, C, N]

        # ---- A_M(ij) = 高斯特征距离 * (IOU + DSim)/2  (Eq.3) ----
        vi = vf.reshape(bn, cn, -1, 1)
        nbr_t = nbr_v.permute(0, 3, 1, 2)                     # [B, C, N, K]
        d_feat = ((vi - nbr_t) ** 2).sum(dim=1)               # [B, N, K]
        gauss = torch.exp(-d_feat / (2 * self.tau ** 2 + 1e-6))

        # DSim：MLP 投影特征余弦相似度
        sem_i = self.mlp_sem(vf)                              # [B, C, N]
        sem_j = self.mlp_sem(nbr_v.reshape(bn, -1, cn).transpose(1, 2))
        sem_j = sem_j.transpose(1, 2).reshape(bn, hn * wn, k, cn)  # [B, N, K, C]
        dsim = F.cosine_similarity(
            sem_i.permute(0, 2, 1).unsqueeze(2),             # [B, N, 1, C]
            sem_j, dim=-1)                                    # [B, N, K]

        # IOU：节点区域空间重叠先验（由窗口偏移预计算）
        offs = _neighbor_offsets(self.window, x.device)
        abs_d = offs.abs()
        overlap = (1 - abs_d[:, 0]).clamp(min=0) * (1 - abs_d[:, 1]).clamp(min=0)
        iou = overlap / (2 - overlap + 1e-6)                  # [K]
        a_m = gauss * ((iou.view(1, 1, -1) + dsim) / 2)       # [B, N, K]

        # ---- 空间可分离注意力 A_Ms (Eq.6) ----
        # 邻居节点 h 向量（复用 mlp_h，GMP 上下文取该邻居自身邻域的最大值简化）
        gmp_n = nbr_v.max(dim=2, keepdim=True).values         # [B, N, 1, C] 节点自身上下文
        concat_n = torch.cat([nbr_v, gmp_n.expand(-1, -1, k, -1)], dim=-1)  # [B, N, K, 2C]
        h_j = self.mlp_h(concat_n.reshape(bn, -1, cn * 2).transpose(1, 2))
        h_j = h_j.transpose(1, 2).reshape(bn, hn * wn, k, cn)  # [B, N, K, C]
        h_i = h.permute(0, 2, 1).unsqueeze(2)                # [B, N, 1, C]
        s_sim = (h_i * h_j).sum(dim=-1) / (cn ** 0.5)         # h_i·h_j^T/sqrt(d)
        a_ms = torch.softmax(s_sim * a_m, dim=-1)             # A_Ms
        agg = (a_ms.unsqueeze(-1) * h_j).sum(dim=2)           # [B, N, C] 空间聚合

        # ---- 通道可分离注意力 A_Mc (Eq.6) ----
        agg_t = agg.transpose(1, 2)                           # [B, C, N]
        c_sim = torch.matmul(agg_t, agg_t.transpose(1, 2)) / (cn ** 0.5)  # [B, C, C]
        a_mc = torch.softmax(c_sim, dim=-1)                   # A_Mc
        gated = torch.matmul(a_mc, agg_t)                     # [B, C, N]

        # ---- 输出 (Eq.6) ----
        out = gated.reshape(bn, cn, hn, wn)
        out = F.interpolate(out, size=(H, W), mode="nearest")  # 恢复到 F_M^0 分辨率
        return x + torch.relu(out)


class HCGSA(nn.Module):
    """高层全局上下文语义注意力（Eq.4 / Eq.7）：挖掘全局语义与长程依赖。

    以通道为图节点：N_H = C_h。
    图节点:  v_Hi = F_H^0(X_i,:,X_i) * CW(i)，CW 为通道注意力权重
    邻接:    A_H(ij) = exp(-||v_Hi-v_Hj||^2/(2 tau_H^2)) * corr(v_Hi, v_Hj)
             corr 为通道间协方差/标准差归一化（线性相关度）
    注意力:  h_Hi = MLP( concat(v_Hi, GAP(V_H)) )
             Q=W_q h_Hi, K=W_k h_Hi, V=W_v h_Hi
             O_H = Softmax(QK^T/sqrt(d) ⊙ A_H(ij))
    聚合:    F_H^(1)(X) = F_H^0(X) + sigma(O_H V)
    """

    def __init__(self, channels, tau=1.0, hidden=None):
        super().__init__()
        self.tau = tau
        hidden = hidden or max(channels // 4, 16)
        # CW(i)：通道注意力权重（SE 风格）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1), nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1), nn.Sigmoid(),
        )
        self.mlp_h = nn.Sequential(
            nn.Conv1d(channels + 1, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
        )
        self.wq = nn.Conv1d(channels, channels, 1)
        self.wk = nn.Conv1d(channels, channels, 1)
        self.wv = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b, c, H, W = x.shape                                  # F_H^0: [B, C_h, H/4, W/4]
        # CW(i) 通道权重 (Eq.4)
        cw = self.se(x)                                       # [B, C, 1, 1]
        v = x * cw
        # 逐通道描述子 d_i = GAP(v_i)，全局上下文 GAP(V_H)（单标量广播）
        d = F.adaptive_avg_pool2d(v, 1).reshape(b, c, 1)      # [B, C, 1]
        g = d.mean(dim=1, keepdim=True)                       # [B, 1, 1] 全局上下文
        h = self.mlp_h(torch.cat([d, g], dim=1))              # h_Hi (Eq.7), [B, C+1, 1]
        q = self.wq(h)
        k = self.wk(h)
        vv = self.wv(h)

        # ---- A_H：高斯距离 * 通道相关性 (Eq.4) ----
        vf = v.reshape(b, c, -1)                              # [B, C, S]
        vc = vf - vf.mean(dim=-1, keepdim=True)
        cov = torch.matmul(vc, vc.transpose(1, 2)) / (H * W + 1e-6)  # [B, C, C]
        std = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2) + 1e-6)
        corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + 1e-6)
        corr = corr.clamp(-1.0, 1.0)                          # 相关系数定义域 [-1,1]
        hs = h.squeeze(-1)                                    # [B, C]
        dh = (hs.unsqueeze(1) - hs.unsqueeze(2)) ** 2         # [B, C, C]
        a_h = torch.exp(-dh / (2 * self.tau ** 2 + 1e-6)) * corr

        # ---- 全局语义注意力 (Eq.7) ----
        qk = torch.matmul(q, k.transpose(1, 2)) / (c ** 0.5)  # [B, C, C]
        o_h = torch.softmax(qk * a_h, dim=-1)                 # O_H
        out = torch.matmul(o_h, vv)                           # O_H V: [B, C, 1]
        return x + torch.relu(out.unsqueeze(-1))              # F_H^(1) (Eq.7)


class GATFE(nn.Module):
    """GAT-FE 模块整体：三级图注意力 + Eq.8 对齐融合。"""

    def __init__(self, low_channels=64, mid_channels=128, high_channels=256,
                 tfic_channels=256, low_patch=2, mid_patch=2, knn_window=3,
                 tau_L=1.0, tau_M=1.0, tau_H=1.0, hidden=128):
        super().__init__()
        self.lgna = LGNABlock(low_channels, patch=low_patch, tau=tau_L,
                              window=knn_window, hidden=hidden)
        self.mcgsa = MCGSABlock(mid_channels, patch=mid_patch, tau=tau_M,
                                window=knn_window, hidden=hidden)
        self.hcgsa = HCGSA(high_channels, tau=tau_H, hidden=hidden)
        # Eq.8 前的通道对齐投影
        self.proj_l = nn.Conv2d(low_channels, tfic_channels, 1)
        self.proj_m = nn.Conv2d(mid_channels, tfic_channels, 1)
        self.proj_h = nn.Conv2d(high_channels, tfic_channels, 1)
        # 三重注意力权重 W（SE 风格，Eq.8）
        self.triple_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(tfic_channels * 3, tfic_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(tfic_channels // 2, tfic_channels * 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, f_l0, f_m0, f_h0):
        f_l1 = self.lgna(f_l0)                                # F_L^(1)
        f_m1 = self.mcgsa(f_m0)                               # F_M^(1)
        f_h1 = self.hcgsa(f_h0)                               # F_H^(1)

        # 缺陷置信度图：低层 w_i 上采样到中层尺度（供 TFIC 的 GCB 加权）
        wmap = self.lgna.mlp_pixel(
            F.avg_pool2d(f_l0, self.lgna.patch, self.lgna.patch)
             .reshape(f_l0.shape[0], f_l0.shape[1], -1)
        ).reshape(f_l0.shape[0], 1,
                  f_l0.shape[2] // self.lgna.patch,
                  f_l0.shape[3] // self.lgna.patch)
        conf = F.interpolate(wmap, scale_factor=2, mode="bilinear", align_corners=False)

        # ---- Eq.8：对齐 + 拼接 + 三重注意力加权（仅模块内部全局加权用） ----
        pl = self.proj_l(f_l1)                                # [B, C, H, W]
        pm_mid = self.proj_m(f_m1)                            # [B, C, H/2, W/2] 保留尺度
        ph_deep = self.proj_h(f_h1)                           # [B, C, H/4, W/4] 保留尺度
        pm = F.interpolate(pm_mid, scale_factor=2, mode="bilinear", align_corners=False)
        ph = F.interpolate(ph_deep, scale_factor=4, mode="bilinear", align_corners=False)
        concat = torch.cat([pl, pm, ph], dim=1)               # [B, 3C, H, W]
        w = self.triple_attn(concat)                          # 三重注意力权重 W
        fused = concat * w
        # 三级独立尺度张量保留（送入 TFIC 做多尺度双向补偿）
        return pl, pm_mid, ph_deep, fused, conf

