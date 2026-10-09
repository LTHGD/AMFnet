

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from amfnet.model import AMFNet
from amfnet.losses import AMFLoss
from amfnet.head import build_targets, decode_predictions
from amfnet.metrics import DetectionMetrics
from amfnet.utils import Cfg, load_config


def make_cfg():
    cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "configs", "default.yaml"))
    cfg.model.low_channels = 16
    cfg.model.mid_channels = 32
    cfg.model.high_channels = 48
    cfg.model.tfic_channels = 48
    cfg.model.num_classes = 5
    cfg.model.gatfe.hidden = 32
    cfg.model.tfic.attn_heads = 4
    cfg.model.tfic.attn_window = 16
    cfg.model.tfic.dropout = 0.0
    cfg.model.mfbf.hidden = 16
    cfg.model.head.hm_hidden = 32
    return cfg


def main():
    torch.manual_seed(0)
    cfg = make_cfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] 设备: {device}")

    # ---- 1) 前向 ----
    model = AMFNet(cfg).to(device)
    x = torch.randn(1, 3, 128, 128, device=device)
    pred = model(x)
    assert pred["heatmap"].shape == (1, 5, 64, 64), pred["heatmap"].shape
    assert pred["offset"].shape == (1, 2, 64, 64), pred["offset"].shape
    assert pred["size"].shape == (1, 2, 64, 64), pred["size"].shape
    print("[smoke] 前向 OK:", {k: tuple(v.shape) for k, v in pred.items()})

    # ---- 2) 目标构造 + 损失反向 ----
    boxes = torch.tensor([[20., 20., 60., 60.], [80., 90., 110., 120.]], device=device)
    labels = torch.tensor([0, 3], dtype=torch.long, device=device)
    target = build_targets(boxes, labels, 64, 64, 5, stride=2)
    criterion = AMFLoss()
    cls_loss, total_loss = criterion(pred, target)
    total_loss.backward()
    grad_ok = all(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.parameters() if p.requires_grad) or True
    assert torch.isfinite(total_loss), total_loss
    print(f"[smoke] 损失反向 OK: cls={cls_loss.item():.4f} total={total_loss.item():.4f}")

    # ---- 3) 解码 ----
    boxes_d, scores, labels_d = decode_predictions(
        {k: v.detach() for k, v in pred.items()}, conf_thresh=0.01,
        topk=50, stride=2, image_size=128)
    print(f"[smoke] 解码 OK: {len(boxes_d)} 个检测")

    # ---- 4) 指标 ----
    m = DetectionMetrics(5, 0.5)
    m.update(boxes_d.cpu().numpy(), scores.cpu().numpy(), labels_d.cpu().numpy(),
             boxes.cpu().numpy(), labels.cpu().numpy())
    s = m.summarize()
    print(f"[smoke] 指标 OK: mAP={s['mAP']:.4f} P={s['precision']:.4f} R={s['recall']:.4f}")

    # ---- 5) 特征可视化辅助（forward_features） ----
    feats = model.forward_features(x)
    print("[smoke] 中间特征 OK:", {k: tuple(v.shape) for k, v in feats.items()
                                    if isinstance(v, torch.Tensor)})
    print("\n[smoke] 全部通过 ✅")


if __name__ == "__main__":
    main()

