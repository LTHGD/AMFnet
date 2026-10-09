
"""检测头：缺陷分类与定位（CenterNet 风格）。

结构：热图头（分类）+ 中心偏移头 + 尺寸回归头，输入为 MFBF 融合特征
（中间尺度 H/2 x W/2）。配合 Focal Loss（分类）与 CIoU Loss（回归），
即论文 Eq.20 的 L_total = L_cls + lambda * L_reg。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import nms


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


def _gaussian_radius(h, w, min_overlap=0.7):
    """CenterNet 风格高斯半径（近似），用于生成热图标签。"""
    import math
    a1 = 1
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0.0))
    r1 = (b1 - sq1) / 2
    return max(0, int(r1))


def build_targets(boxes, labels, out_h, out_w, num_classes, stride=2):
    """由标注框生成检测目标。

    boxes: [N, 4] xyxy（输入图像像素坐标）; labels: [N] 类别 id
    返回 dict: heatmap [1, num_classes, out_h, out_w]
               offset  [1, 2, out_h, out_w]
               size    [1, 2, out_h, out_w]
    """
    hm = torch.zeros(1, num_classes, out_h, out_w, device=boxes.device)
    offset = torch.zeros(1, 2, out_h, out_w, device=boxes.device)
    size = torch.zeros(1, 2, out_h, out_w, device=boxes.device)
    if boxes.numel() == 0:
        return {"heatmap": hm, "offset": offset, "size": size}

    # 缩放框到输出特征尺度
    bx = boxes.clone()
    bx[:, [0, 2]] = bx[:, [0, 2]] / stride
    bx[:, [1, 3]] = bx[:, [1, 3]] / stride
    cx = (bx[:, 0] + bx[:, 2]) / 2
    cy = (bx[:, 1] + bx[:, 3]) / 2
    w = bx[:, 2] - bx[:, 0]
    h = bx[:, 3] - bx[:, 1]

    ct = torch.stack([cx, cy], dim=1).round().long()           # 中心整数坐标
    for i in range(len(labels)):
        cl = int(labels[i].item())
        x, y = ct[i, 0].item(), ct[i, 1].item()
        if not (0 <= x < out_w and 0 <= y < out_h):
            continue
        radius = _gaussian_radius(h[i].item(), w[i].item())
        radius = max(radius, 1)
        # 高斯核写入热图
        xx = torch.arange(out_w, device=boxes.device, dtype=torch.float32)
        yy = torch.arange(out_h, device=boxes.device, dtype=torch.float32)
        gx = torch.exp(-((xx - x) ** 2) / (2 * radius ** 2))
        gy = torch.exp(-((yy - y) ** 2) / (2 * radius ** 2))
        g = torch.outer(gy, gx)                                 # [out_h, out_w]
        hm[0, cl] = torch.maximum(hm[0, cl], g)
        # 中心点：offset = 中心 - 整数坐标（取整误差回归）
        offset[0, 0, y, x] = cx[i] - x
        offset[0, 1, y, x] = cy[i] - y
        size[0, 0, y, x] = w[i]
        size[0, 1, y, x] = h[i]
    return {"heatmap": hm, "offset": offset, "size": size}


class CenterNetHead(nn.Module):
    """CenterNet 风格检测头。"""

    def __init__(self, in_channels, num_classes, hm_hidden=128):
        super().__init__()
        self.num_classes = num_classes
        # 分类热图头
        self.hm_head = nn.Sequential(
            ConvBlock(in_channels, hm_hidden, stride=1, kernel_size=3),
            nn.Conv2d(hm_hidden, num_classes, 1),
        )
        # 中心偏移头
        self.offset_head = nn.Sequential(
            ConvBlock(in_channels, hm_hidden, stride=1, kernel_size=3),
            nn.Conv2d(hm_hidden, 2, 1),
        )
        # 尺寸回归头
        self.size_head = nn.Sequential(
            ConvBlock(in_channels, hm_hidden, stride=1, kernel_size=3),
            nn.Conv2d(hm_hidden, 2, 1),
        )

    def forward(self, x):
        return {
            "heatmap": self.hm_head(x),
            "offset": self.offset_head(x),
            "size": self.size_head(x),
        }


def decode_predictions(pred, conf_thresh=0.1, topk=200, stride=2,
                       image_size=None, nms_thresh=0.5):
    """解码热图预测为检测框。

    pred: dict{heatmap, offset, size}（尺度为输出特征图）
    返回: boxes [M, 4]（xyxy，原图坐标）, scores [M], labels [M] (int)
    """
    hm = torch.sigmoid(pred["heatmap"][0])                     # [C, H, W]
    offset = pred["offset"][0]                                 # [2, H, W]
    size = pred["size"][0]                                     # [2, H, W]
    c, h, w = hm.shape

    # 3x3 局部峰值抑制
    peak = F.max_pool2d(hm.unsqueeze(0), kernel_size=3, stride=1, padding=1)
    keep = (hm == peak.squeeze(0)) & (hm >= conf_thresh)       # [C, H, W]

    scores, labels, ys, xs = [], [], [], []
    for cl in range(c):
        idx = keep[cl].nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        sc = hm[cl][idx[:, 0], idx[:, 1]]
        top = sc.argsort(descending=True)[:topk]
        labels.append(torch.full((len(top),), cl, dtype=torch.long, device=hm.device))
        scores.append(sc[top])
        ys.append(idx[top, 0])
        xs.append(idx[top, 1])
    if not scores:
        return (torch.zeros(0, 4, device=hm.device),
                torch.zeros(0, device=hm.device),
                torch.zeros(0, dtype=torch.long, device=hm.device))
    scores = torch.cat(scores)
    labels = torch.cat(labels)
    ys = torch.cat(ys)
    xs = torch.cat(xs)

    # 中心 + 偏移 + 尺寸 -> xyxy（原图坐标）
    cx = xs + offset[0, ys, xs]
    cy = ys + offset[1, ys, xs]
    bw = size[0, ys, xs]
    bh = size[1, ys, xs]
    boxes = torch.stack([
        (cx - bw / 2) * stride, (cy - bh / 2) * stride,
        (cx + bw / 2) * stride, (cy + bh / 2) * stride,
    ], dim=1)

    # 越界裁剪
    if image_size is not None:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image_size)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image_size)

    # NMS（按类别独立做）
    final_b, final_s, final_l = [], [], []
    for cl in range(c):
        m = labels == cl
        if m.sum() == 0:
            continue
        idx = nms(boxes[m], scores[m], nms_thresh)
        final_b.append(boxes[m][idx])
        final_s.append(scores[m][idx])
        final_l.append(torch.full((len(idx),), cl, dtype=torch.long, device=hm.device))
    if not final_b:
        return (torch.zeros(0, 4, device=hm.device),
                torch.zeros(0, device=hm.device),
                torch.zeros(0, dtype=torch.long, device=hm.device))
    return torch.cat(final_b), torch.cat(final_s), torch.cat(final_l)
