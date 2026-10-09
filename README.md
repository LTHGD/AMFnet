# AMFNet 官方实现（Python / PyTorch）

论文：《Information Compensation-based Adaptive Multi-scale Fusion Network
for Chip Small Defect Detection》（IEEE TSM）
对应论文 Fig.1：`Backbone(Eq.1) → GAT-FE(Eq.2-8) → TFIC(Eq.9-12) → MFBF(Eq.13-16) → Detection Head`


## 安装与运行

```bash
pip install -r requirements.txt

# 1) 冒烟测试（无需真实数据，验证前向/反向可跑通）
python scripts/smoke_test.py

# 2) 合成数据快速训练（验证全流程）
python train.py --synthetic --epochs 2 --image-size 128 --batch-size 2

# 3) 真实数据训练（论文协议：8:1:1 分层划分，5 个 seed 各 150 epochs）
python train.py --config configs/default.yaml --trials

# 4) 评测（Mean±Std / 95% CI）
python eval.py --config configs/default.yaml --ckpt-dir outputs

# 5) 单图推理
python inference.py --ckpt outputs/seed10/best.pt --image test.png --out result.png
```

## 数据准备

- **Automotive MEMS 压力传感器**（5 类，5874 张，Table II）：
  `data/mems/JPEGImages/*.jpg` + `data/mems/Annotations/*.xml`（VOC 格式）
- **PCB**（6 类，693 张）：`data/pcb/JPEGImages/*.jpg` + `data/pcb/Annotations/*.xml`

在 `configs/default.yaml` 中切换 `data.dataset: mems | pcb`，并设置
`data.data_root` 与 `model.num_classes`（mems=5, pcb=6）。
