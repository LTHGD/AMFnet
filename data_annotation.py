"""数据加载（论文 Section III-A）。

支持两个数据集（Table II）：
  - Automotive MEMS pressure sensor（FIRC 采集，5 类，5874 张）
  - PCB（Ding et al. 公开，6 类，693 张）

标注格式：
  - voc ：data_root/JPEGImages/*.jpg + data_root/Annotations/*.xml
  - coco：data_root/<coco_json>（COCO 实例分割/检测标注）
划分：按类别分层的 8:1:1 随机划分（论文 5 个 seed 各做一次）；
增强：随机水平/垂直翻转、亮度/对比度（仅训练集）。
另提供 SyntheticDefectDataset，用于无真实数据时验证流程。
"""

import os
import glob
import random
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# 论文 Fig.6 / Fig.7 类别定义
MEMS_CLASSES = [
    "aluminum_wire_bonding_broken", "chip_injury", "chip_scratch",
    "glue_surface_wrinkle", "gold_wire_bonding_broken",
]
PCB_CLASSES = [
    "Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper",
]

DATASET_CLASSES = {"mems": MEMS_CLASSES, "pcb": PCB_CLASSES}


def _read_voc_annotations(ann_path, class_to_id):
    """解析 VOC XML 标注 -> (boxes [M,4] xyxy, labels [M])。"""
    tree = ET.parse(ann_path)
    root = tree.getroot()
    size = root.find("size")
    w_img, h_img = int(size.find("width").text), int(size.find("height").text)
    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in class_to_id:
            continue
        bnd = obj.find("bndbox")
        x1 = float(bnd.find("xmin").text)
        y1 = float(bnd.find("ymin").text)
        x2 = float(bnd.find("xmax").text)
        y2 = float(bnd.find("ymax").text)
        boxes.append([x1, y1, x2, y2])
        labels.append(class_to_id[name])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), \
        np.array(labels, dtype=np.int64).reshape(-1)


def _read_coco_annotations(img_id, coco_root, json_name, class_to_id):
    """读取单张图的 COCO 标注（首次调用时加载整个 json 到缓存）。"""
    import json
    cache = _read_coco_annotations.__dict__.setdefault("_cache", {})
    key = os.path.join(coco_root, json_name)
    if key not in cache:
        with open(key, "r", encoding="utf-8") as f:
            cache[key] = json.load(f)
    data = cache[key]
    boxes, labels = [], []
    for ann in data["annotations"]:
        if ann["image_id"] != img_id:
            continue
        cat_id = ann["category_id"]
        if cat_id not in class_to_id:
            continue
        x, y, w, h = ann["bbox"]                              # COCO: x,y,w,h
        boxes.append([x, y, x + w, y + h])
        labels.append(class_to_id[cat_id])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), \
        np.array(labels, dtype=np.int64).reshape(-1)


def stratified_split(samples, ratio=(0.8, 0.1, 0.1), seed=0):
    """按类别分层随机划分（8:1:1），保证各类别分布一致。

    samples: [(img_path, ann_path_or_id), ...]
    """
    rng = random.Random(seed)
    # 按类别聚合样本（一个样本可能含多类，取其主类别）
    buckets = defaultdict(list)
    for s in samples:
        buckets[s[2]].append(s)                               # s[2] 为主类别 id
    train, val, test = [], [], []
    for cls, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(round(n * ratio[0]))
        n_va = int(round(n * ratio[1]))
        train.extend(items[:n_tr])
        val.extend(items[n_tr:n_tr + n_va])
        test.extend(items[n_tr + n_va:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _scale_box(box, img_size, target_size):
    """把原图坐标框缩放到 target_size 尺度。"""
    h, w = img_size
    th, tw = target_size
    sx, sy = tw / w, th / h
    box = box.copy()
    box[:, [0, 2]] *= sx
    box[:, [1, 3]] *= sy
    return box


class DefectDataset(Dataset):
    """通用缺陷检测数据集（VOC / COCO）。"""

    def __init__(self, root, classes, ann_format="voc", split="train",
                 image_size=640, transform=None, seed=0, coco_json="annotations/instances.json"):
        super().__init__()
        self.root = root
        self.classes = list(classes)
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        self.ann_format = ann_format
        self.image_size = (image_size, image_size)
        self.transform = transform
        self.samples = self._scan(seed, split, coco_json)
        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"数据集为空：{root}（{ann_format}, split={split}）。"
                f"请确认目录结构为 voc: <root>/JPEGImages + <root>/Annotations，"
                f"或 coco: <root>/{coco_json}")

    # ---------- 扫描与划分 ----------
    def _scan(self, seed, split, coco_json):
        if self.ann_format == "voc":
            imgs = sorted(glob.glob(os.path.join(self.root, "JPEGImages", "*")))
            samples = []
            for p in imgs:
                stem = os.path.splitext(os.path.basename(p))[0]
                ann = os.path.join(self.root, "Annotations", stem + ".xml")
                if not os.path.exists(ann):
                    continue
                boxes, labels = _read_voc_annotations(ann, self.class_to_id)
                main_cls = int(labels[0]) if len(labels) else -1
                samples.append((p, ann, main_cls))
            samples = [s for s in samples if s[2] >= 0]
        elif self.ann_format == "coco":
            import json
            with open(os.path.join(self.root, coco_json), "r", encoding="utf-8") as f:
                coco = json.load(f)
            img_dir = os.path.join(self.root, os.path.dirname(coco_json))
            samples = []
            for im in coco["images"]:
                p = os.path.join(img_dir, im["file_name"])
                if not os.path.exists(p):
                    continue
                boxes, labels = _read_coco_annotations(im["id"], self.root, coco_json,
                                                       self.class_to_id)
                main_cls = int(labels[0]) if len(labels) else -1
                samples.append((p, im["id"], main_cls))
            samples = [s for s in samples if s[2] >= 0]
        else:
            raise ValueError(f"未知标注格式: {self.ann_format}")
        train, val, test = stratified_split(samples, seed=seed)
        return {"train": train, "val": val, "test": test}[split]

    # ---------- 迭代 ----------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, ann, _ = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size
        if self.ann_format == "voc":
            boxes, labels = _read_voc_annotations(ann, self.class_to_id)
        else:
            boxes, labels = _read_coco_annotations(ann, self.root, "", self.class_to_id)
        # 统一缩放
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        boxes = _scale_box(boxes, (h0, w0), self.image_size)
        # 增强（仅训练）
        if self.transform is not None:
            img, boxes = self.transform(img, boxes)
        x = T.ToTensor()(img)
        return x, {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels)}


# ---------- 训练增强（水平/垂直翻转、亮度对比度） ----------
class TrainTransform:
    def __init__(self, hflip=0.5, vflip=0.5, brightness=0.3, contrast=0.3):
        self.hflip = hflip
        self.vflip = vflip
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img, boxes):
        w, h = img.size
        if random.random() < self.hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        if random.random() < self.vflip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        if random.random() < 0.5:
            img = T.ColorJitter(brightness=self.brightness, contrast=self.contrast)(img)
        return img, boxes


def make_transforms(train=True, cfg=None):
    if not train:
        return None
    cfg = cfg or {}
    return TrainTransform(
        hflip=0.5, vflip=0.5,
        brightness=cfg.get("brightness", 0.3), contrast=cfg.get("contrast", 0.3),
    )


def build_loaders(cfg, seed=0, split_ratio=(0.8, 0.1, 0.1)):
    """按配置构建 train/val/test DataLoader（分层 8:1:1）。"""
    d = cfg.data
    classes = DATASET_CLASSES[d.dataset] if d.dataset in DATASET_CLASSES \
        else d.get("mems_classes" if d.dataset == "mems" else "pcb_classes")
    image_size = int(d.image_size)
    root = d.data_root

    ds_kwargs = dict(classes=classes, ann_format=d.ann_format,
                     image_size=image_size, seed=seed,
                     coco_json=d.get("coco_json", "annotations/instances.json"))
    train_ds = DefectDataset(root, split="train", transform=make_transforms(True), **ds_kwargs)
    val_ds = DefectDataset(root, split="val", transform=None, **ds_kwargs)
    test_ds = DefectDataset(root, split="test", transform=None, **ds_kwargs)

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        max_n = max(len(b[1]["boxes"]) for b in batch)
        boxes = torch.zeros(len(batch), max_n, 4)
        labels = torch.full((len(batch), max_n), -1, dtype=torch.long)
        for i, (_, t) in enumerate(batch):
            n = len(t["boxes"])
            boxes[i, :n] = t["boxes"]
            labels[i, :n] = t["labels"]
        return imgs, {"boxes": boxes, "labels": labels}

    kwargs = dict(batch_size=int(d.batch_size), num_workers=int(d.num_workers),
                  collate_fn=collate, drop_last=False)
    return (torch.utils.data.DataLoader(train_ds, shuffle=True, **kwargs),
            torch.utils.data.DataLoader(val_ds, shuffle=False, **kwargs),
            torch.utils.data.DataLoader(test_ds, shuffle=False, **kwargs))


# ---------- 合成数据集（冒烟测试 / 流程验证用） ----------
class SyntheticDefectDataset(Dataset):
    """随机生成带矩形缺陷框的噪声图像，用于无真实数据时验证训练流程。"""

    def __init__(self, num_samples=200, image_size=640, num_classes=5,
                 seed=0, max_boxes=3):
        rng = np.random.RandomState(seed)
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        self._rng = rng

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        s = self.image_size
        img = np.zeros((s, s, 3), dtype=np.uint8)
        boxes, labels = [], []
        rng = np.random.RandomState(hash((i, 42)) % (2 ** 32))
        for _ in range(int(rng.randint(1, self.max_boxes + 1))):
            w = int(rng.randint(s // 40, s // 10))
            h = int(rng.randint(s // 40, s // 10))
            x1 = int(rng.randint(0, s - w))
            y1 = int(rng.randint(0, s - h))
            v = int(rng.randint(120, 255))
            img[y1:y1 + h, x1:x1 + w] = (v, v // 2, v // 3)
            boxes.append([x1, y1, x1 + w, y1 + h])
            labels.append(int(rng.randint(0, self.num_classes)))
        x = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return x, {"boxes": torch.tensor(boxes, dtype=torch.float32),
                   "labels": torch.tensor(labels, dtype=torch.long)}
