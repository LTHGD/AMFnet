"""损失函数（论文 Eq.20）。

    L_total = L_cls + lambda * L_reg,   lambda = 0.5

  - L_cls：Focal Loss（缓解类别不平衡），作用于中心热图分类
  - L_reg：CIoU Loss（边界框回归），作用于中心偏移 + 尺寸
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """CenterNet 风格 Focal Loss（分类，Eq.20 的 L_cls）。

    正样本：(1-p)^alpha * log(p)；负样本由高斯标注惩罚项加权。
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """pred: [B, C, H, W] 未归一化 logits; target: [B, C, H, W] 高斯热图。"""
        pos = (target == 1).float()
        neg = (target < 1).float()
        p = torch.sigmoid(pred)
        eps = 1e-6
        # 正样本
        pos_loss = -torch.pow(1 - p, self.alpha) * torch.log(p + eps) * pos
        # 负样本（高斯惩罚：靠近目标中心权重降低）
        neg_loss = -(torch.pow(1 - target, self.beta) * torch.pow(p, self.alpha)
                     * torch.log(1 - p + eps) * neg)
        num_pos = pos.sum().clamp(min=1)
        return (pos_loss.sum() + neg_loss.sum()) / num_pos


def _ciou_loss(pred_boxes, target_boxes):
    """CIoU Loss（论文 Eq.20 的 L_reg），输入均为 cx,cy,w,h。"""
    p = pred_boxes
    t = target_boxes

    pcx, pcy, pw, ph = p.unbind(-1)
    tcx, tcy, tw, th = t.unbind(-1)
    p_x1, p_y1, p_x2, p_y2 = pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2
    t_x1, t_y1, t_x2, t_y2 = tcx - tw / 2, tcy - th / 2, tcx + tw / 2, tcy + th / 2

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_p = pw * ph
    area_t = tw * th
    union = area_p + area_t - inter + 1e-6
    iou = inter / union

    # 中心点距离
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    # 最小闭包框对角线
    c_x1 = torch.min(p_x1, t_x1)
    c_y1 = torch.min(p_y1, t_y1)
    c_x2 = torch.max(p_x2, t_x2)
    c_y2 = torch.max(p_y2, t_y2)
    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + 1e-6

    # 宽高比一致性项
    pi = 4 * torch.atan(tw / (th + 1e-6)) - 4 * torch.atan(pw / (ph + 1e-6))
    v = (pi / torch.pi) ** 2
    alpha = v / (1 - iou + v + 1e-6)

    return (1 - iou + rho2 / c2 + alpha * v).mean()


class CIoULoss(nn.Module):
    """CIoU 边界框回归损失。"""

    def forward(self, pred_boxes, target_boxes):
        return _ciou_loss(pred_boxes, target_boxes)


class AMFLoss(nn.Module):
    """AMFNet 总损失：L_total = L_cls + lambda * L_reg（Eq.20）。"""

    def __init__(self, focal_alpha=2.0, focal_beta=4.0, lambda_reg=0.5):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.focal = FocalLoss(alpha=focal_alpha, beta=focal_beta)

    def forward(self, pred, target):
        """pred / target 均为 dict{heatmap, offset, size}（B 张图拼接的标注）。"""
        cls_loss = self.focal(pred["heatmap"], target["heatmap"])
        # 回归损失仅在热图中心点位置计算
        pos_mask = (target["heatmap"] == 1).float()            # [B, C, H, W]
        # 取每个中心点的 offset/size 与对应类别位置
        b, c, h, w = target["heatmap"].shape
        if pos_mask.sum() < 1:
            return cls_loss, cls_loss + 0.0 * self.lambda_reg
        idx = pos_mask.nonzero()                               # [M, 4] (b, cl, y, x)
        bs, ys, xs = idx[:, 0], idx[:, 2], idx[:, 3]
        p_off = pred["offset"][bs, :, ys, xs]                  # [M, 2]
        t_off = target["offset"][bs, :, ys, xs]
        p_size = pred["size"][bs, :, ys, xs]                   # [M, 2]
        t_size = target["size"][bs, :, ys, xs]
        # 由中心 + 偏移 + 尺寸构造 cxcywh 框
        pcx = xs.float() + p_off[:, 0]
        pcy = ys.float() + p_off[:, 1]
        tcx = xs.float() + t_off[:, 0]
        tcy = ys.float() + t_off[:, 1]
        p_boxes = torch.stack([pcx, pcy, p_size[:, 0], p_size[:, 1]], dim=1)
        t_boxes = torch.stack([tcx, tcy, t_size[:, 0], t_size[:, 1]], dim=1)
        reg_loss = _ciou_loss(p_boxes, t_boxes)
        return cls_loss, cls_loss + self.lambda_reg * reg_loss

