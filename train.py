import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F

from amfnet.model import AMFNet
from amfnet.losses import AMFLoss
from amfnet.head import build_targets, decode_predictions
from amfnet.metrics import DetectionMetrics
from amfnet.utils import (Cfg, load_config, set_seed, resolve_device,
                          save_checkpoint)
from amfnet.data import (SyntheticDefectDataset, build_loaders,
                         DATASET_CLASSES)


def parse_args():
    p = argparse.ArgumentParser(description="AMFNet 训练")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=None, help="覆盖配置中的 seed")
    p.add_argument("--trials", action="store_true", help="按 seed_list 依次训练 5 次")
    p.add_argument("--epochs", type=int, default=None, help="覆盖训练 epoch 数")
    p.add_argument("--image-size", type=int, default=None, help="覆盖输入分辨率")
    p.add_argument("--batch-size", type=int, default=None, help="覆盖 batch size")
    p.add_argument("--synthetic", action="store_true", help="使用合成数据（冒烟测试）")
    p.add_argument("--num-classes", type=int, default=5, help="合成数据类别数")
    p.add_argument("--out", default=None, help="覆盖输出目录")
    return p.parse_args()


def build_model_and_loaders(cfg, args, seed, device):
    if args.synthetic:
        from torch.utils.data import DataLoader
        n = 200
        cfg.model.num_classes = args.num_classes
        cfg.data.image_size = args.image_size or cfg.data.image_size
        tr = SyntheticDefectDataset(num_samples=n, image_size=int(cfg.data.image_size),
                                    num_classes=args.num_classes, seed=seed)
        va = SyntheticDefectDataset(num_samples=max(n // 10, 10),
                                    image_size=int(cfg.data.image_size),
                                    num_classes=args.num_classes, seed=seed + 1)

        def collate(batch):
            imgs = torch.stack([b[0] for b in batch])
            max_n = max(len(b[1]["boxes"]) for b in batch)
            boxes = torch.zeros(len(batch), max_n, 4)
            labels = torch.full((len(batch), max_n), -1, dtype=torch.long)
            for i, (_, t) in enumerate(batch):
                m = len(t["boxes"])
                boxes[i, :m] = t["boxes"]
                labels[i, :m] = t["labels"]
            return imgs, {"boxes": boxes, "labels": labels}

        bs = args.batch_size or int(cfg.data.batch_size)
        train_loader = DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(va, batch_size=bs, shuffle=False, collate_fn=collate)
        test_loader = val_loader
    else:
        if args.image_size:
            cfg.data.image_size = args.image_size
        if args.batch_size:
            cfg.data.batch_size = args.batch_size
        train_loader, val_loader, test_loader = build_loaders(cfg, seed=seed)
    model = AMFNet(cfg).to(device)
    return model, train_loader, val_loader, test_loader


def evaluate(model, loader, device, cfg, image_size):
    """验证集评测：解码 + IoU 匹配 -> Precision / Recall / mAP。"""
    model.eval()
    metrics = DetectionMetrics(int(cfg.model.num_classes),
                               iou_thresh=float(cfg.train.iou_thresh))
    out_stride = 2
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            pred = model(imgs)
            conf = float(cfg.eval.conf_thresh)
            topk = int(cfg.eval.topk)
            for i in range(imgs.shape[0]):
                p = {k: v[i:i + 1] for k, v in pred.items()}
                boxes, scores, labels = decode_predictions(
                    p, conf_thresh=conf, topk=topk, stride=out_stride,
                    image_size=image_size, nms_thresh=float(cfg.train.iou_thresh))
                gt_n = (targets["labels"][i] >= 0).sum().item()
                metrics.update(
                    boxes.cpu().numpy(), scores.cpu().numpy(),
                    labels.cpu().numpy(),
                    targets["boxes"][i, :gt_n].numpy(),
                    targets["labels"][i, :gt_n].numpy())
    return metrics.summarize()


def run_trial(cfg, args, seed, device):
    """单个 seed 的完整训练 + 验证。返回 (val_metrics, best_ckpt_path)。"""
    set_seed(seed)
    model, train_loader, val_loader, test_loader = build_model_and_loaders(
        cfg, args, seed, device)
    image_size = int(cfg.data.image_size)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(cfg.train.lr),
                                  weight_decay=float(cfg.train.weight_decay))
    epochs = args.epochs or int(cfg.train.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = AMFLoss(focal_alpha=float(cfg.train.focal_alpha),
                        focal_beta=float(cfg.train.focal_beta),
                        lambda_reg=float(cfg.train.lambda_reg))

    out_dir = args.out or os.path.join(cfg.experiment.output_dir, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "train_log.csv")
    logf = open(csv_path, "w", newline="")
    writer = csv.writer(logf)
    writer.writerow(["epoch", "lr", "loss_cls", "loss_reg", "loss_total",
                     "val_mAP", "val_precision", "val_recall"])

    best_map, best_ckpt = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot_cls = tot_reg = tot_n = 0
        for step, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            # 构造热图目标
            out_h, out_w = imgs.shape[2] // 2, imgs.shape[3] // 2
            hm = torch.zeros(imgs.shape[0], int(cfg.model.num_classes),
                             out_h, out_w, device=device)
            off = torch.zeros(imgs.shape[0], 2, out_h, out_w, device=device)
            sz = torch.zeros(imgs.shape[0], 2, out_h, out_w, device=device)
            for i in range(imgs.shape[0]):
                gt_n = (targets["labels"][i] >= 0).sum().item()
                if gt_n == 0:
                    continue
                t = build_targets(targets["boxes"][i, :gt_n],
                                  targets["labels"][i, :gt_n],
                                  out_h, out_w, int(cfg.model.num_classes), stride=2)
                hm[i] = t["heatmap"][0]
                off[i] = t["offset"][0]
                sz[i] = t["size"][0]
            target = {"heatmap": hm, "offset": off, "size": sz}

            pred = model(imgs)
            cls_loss, total_loss = criterion(pred, target)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            tot_cls += cls_loss.item()
            tot_reg += (total_loss.item() - cls_loss.item())
            tot_n += 1
            if (step + 1) % int(cfg.train.log_interval) == 0:
                print(f"  [seed{seed} ep{epoch} step{step+1}] "
                      f"cls={cls_loss.item():.4f} total={total_loss.item():.4f}")
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        val = evaluate(model, val_loader, device, cfg, image_size)
        writer.writerow([epoch, f"{lr:.2e}", f"{tot_cls/max(tot_n,1):.4f}",
                         f"{tot_reg/max(tot_n,1):.4f}",
                         f"{(tot_cls+tot_reg)/max(tot_n,1):.4f}",
                         f"{val['mAP']:.4f}", f"{val['precision']:.4f}",
                         f"{val['recall']:.4f}"])
        logf.flush()
        print(f"[seed{seed} ep{epoch}] lr={lr:.2e} "
              f"loss_cls={tot_cls/max(tot_n,1):.4f} "
              f"val_mAP={val['mAP']:.4f} val_P={val['precision']:.4f} "
              f"val_R={val['recall']:.4f} ({time.time()-t0:.1f}s)")

        if val["mAP"] > best_map:
            best_map = val["mAP"]
            best_ckpt = os.path.join(out_dir, "best.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, cfg, best_ckpt)
        if epoch % int(cfg.train.save_interval) == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, cfg,
                            os.path.join(out_dir, f"epoch{epoch}.pt"))
    logf.close()
    return {"seed": seed, "val_mAP": best_map, "best_ckpt": best_ckpt}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.experiment.device)
    print(f"设备: {device}")
    seeds = cfg.experiment.seed_list if args.trials else [args.seed or cfg.experiment.seed]
    for seed in seeds:
        print(f"===== 训练试验 seed={seed} =====")
        run_trial(cfg, args, int(seed), device)


if __name__ == "__main__":
    main()
