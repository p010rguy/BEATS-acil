# BEATs-ACIL (ESC-50)

基于 BEATs 的 ESC-50 训练与增量学习实验项目。

## 功能

- 自动下载并解压 ESC-50 数据集
- 生成 25(base) + 5x5(incremental) 的类别划分 `esc50_25_5x5_splits.json`
- Base 阶段监督训练（保存微调模型与 backbone）
- Incremental 阶段 ACIL 风格对齐（可选 Feature Expansion）

## 目录结构

- `main.py`: 训练与增量主入口
- `cli_args.py`: 命令行参数
- `data_utils.py`: 数据集与 `pad_collate`
- `model_utils.py`: BEATs + 分类头封装
- `align_utils.py`: CLS/IL 对齐与评估
- `downloader.py`: ESC-50 下载与解压
- `split_esc50.py`: 划分生成

## 环境要求

- Python `>=3.11`
- 依赖由 `pyproject.toml` / `uv.lock` 管理

安装（uv）：

```bash
uv sync
```

## 数据与预训练权重

程序会自动下载 ESC-50 到默认目录 `Dataset/`。

请确认以下预训练权重存在（默认路径）：

- `checkpoints/BEATs_iter3_plus_AS2M.pt`

如果缺失，base 训练会提示 checkpoint not found。

## 运行方式

### 1) 只做 base 训练

```bash
uv run main.py --basetraining --device mps
```

输出（默认）：

- `checkpoints/beats_finetuned_base25.pt`
- `checkpoints/beats_base25_backbone.pt`

### 2) 只做 incremental（基于已保存 backbone）

```bash
uv run main.py --incremental --device mps
```

输出（默认）：

- `checkpoints/beats_base25_incremental.pt`

### 3) base + incremental 连续执行

```bash
uv run main.py --basetraining --incremental --device mps
```

## 常用参数

- `--dest`: 数据目录（默认 `Dataset`）
- `--force`: 强制重新下载/解压
- `--phase`: 增量阶段数（默认 `5`）
- `--baseclass`: base 类别数（默认 `25`）
- `--resume`: 增量阶段恢复 checkpoint 路径
- `--device`: `cuda|mps|cpu`
- `--fe-dim`: Feature Expansion 维度，默认 `None`（关闭）
- `--rg`: CLS 对齐的岭正则系数（默认 `1e-3`）

示例（启用 FE）：

```bash
uv run main.py --incremental --fe-dim 2000 --rg 1e-3 --device mps
```

## 结果解读

训练日志中常见字段：

- `loss`: 当前 epoch 平均训练损失
- `val_acc`: 验证集准确率
- `Base phase acc`: base 阶段准确率
- `forget`: 遗忘指标（base 初始准确率 - 当前 base 准确率）


