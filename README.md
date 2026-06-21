# Dive-into-e2e-auto-driving
动手学习端到端自动驾驶模型

<details>
<summary>Week 1：基于 MLP 的轨迹预测入门</summary>

## Week 1：基于 MLP 的轨迹预测入门

`week1` 是一个最小可运行的端到端轨迹预测练习：使用一段历史自车轨迹作为输入，预测未来一段时间的二维轨迹。当前实现使用合成数据集和一个简单的 MLP 模型，适合先理解数据读取、模型前向传播、loss、训练循环、验证指标和 checkpoint 保存流程。

### 目录结构与文件目的

```text
week1/
├── train.py
├── eval.py
├── data/
│   ├── dataset.py
│   ├── generate_synthetic_trajectory_dataset.py
│   └── synthetic_trajectory_dataset_v1/
├── models/
│   └── mlp.py
├── losses/
│   └── traj_loss.py
└── utils/
    └── metrics.py
```

`train.py`：训练入口。负责创建数据集和 `DataLoader`，初始化 `TrajMLP`、`TrajLoss` 和 `AdamW` 优化器，按 epoch 执行训练和验证，并在验证集 ADE 变好时保存 `best_model.pt`。

`eval.py`：评估与可视化入口。负责加载训练好的 checkpoint，在测试集上计算 ADE/FDE，并把若干条样本的历史轨迹、真实未来轨迹和预测未来轨迹画成图片。

`data/generate_synthetic_trajectory_dataset.py`：合成数据生成脚本。生成包含直行、左转、右转、加速、减速 5 类运动模式的轨迹数据，并保存为 `.npz` 数据文件、`metadata.csv` 元信息和 `schema.json` 数据说明。

`data/dataset.py`：PyTorch 数据集封装。`MLPDataset` 从指定 split 的 `trajectories.npz` 中读取 `history`、`future`、`label`，并在 `__getitem__` 中返回模型训练需要的张量字典。

`data/synthetic_trajectory_dataset_v1/`：已经生成好的示例数据集。包含 `train` 和 `test` 两个 split，每个 split 下有 `trajectories.npz` 和 `metadata.csv`。

`models/mlp.py`：模型定义。`TrajMLP` 将形状为 `[batch_size, history_len, 2]` 的历史轨迹展平成向量，经过多层全连接网络后输出 `[batch_size, future_len, 2]` 的未来轨迹。

`losses/traj_loss.py`：轨迹预测损失。`TrajLoss` 当前支持 `mse` 和 `smooth_l1` 两种 loss，训练脚本中默认使用 `smooth_l1`。

`utils/metrics.py`：评估指标。`ade` 计算所有未来时间点的平均欧氏距离误差，`fde` 计算最后一个未来点的欧氏距离误差。

### 数据格式

默认数据集路径：

```text
week1/data/synthetic_trajectory_dataset_v1
```

每个样本包含：

```text
history: [10, 2]，历史轨迹点，单位为米
future:  [20, 2]，未来轨迹点，单位为米
label:   轨迹类型编号
```

`label` 与类别名称的对应关系：

```text
0: straight
1: left_turn
2: right_turn
3: accelerate
4: decelerate
```

### 使用方法

以下命令默认从 `week1` 目录执行：

```bash
cd Dive-into-e2e-auto-driving/week1
```

如需重新生成合成数据集：

```bash
python data/generate_synthetic_trajectory_dataset.py \
  --out data/synthetic_trajectory_dataset_v1 \
  --train-samples 10000 \
  --test-samples 2000
```

训练模型：

```bash
python train.py \
  --data-root data/synthetic_trajectory_dataset_v1 \
  --batch-size 256 \
  --epochs 100 \
  --lr 1e-3 \
  --save_dir outputs/checkpoints
```

训练过程中会打印每个 epoch 的训练集和验证集指标：

```text
train_loss, train_ADE, train_FDE, val_loss, val_ADE, val_FDE
```

当验证集 `val_ADE` 刷新最好结果时，脚本会保存：

```text
outputs/checkpoints/best_model.pt
```

checkpoint 中包含：

```text
epoch: 保存时的 epoch
model_state_dict: 模型参数
optimizer_state_dict: 优化器状态
best_val_ade: 当前最好的验证集 ADE
args: 本次训练使用的命令行参数
```

评估并可视化预测结果：

```bash
python eval.py \
  --data-root data/synthetic_trajectory_dataset_v1 \
  --checkpoint outputs/checkpoints/best_model.pt \
  --save-dir outputs/figures \
  --batch-size 256 \
  --num-vis 20
```

评估脚本会输出测试集 ADE/FDE，并在 `outputs/figures` 下保存若干张轨迹对比图。图中：

```text
gray: 历史轨迹
green: 真实未来轨迹
red: 模型预测未来轨迹
black: 当前时刻位置
```

### 关键概念

`model.train()` 用于把模型切换到训练模式，通常放在训练循环开始处。

`model.eval()` 用于把模型切换到评估模式，通常放在验证或测试前。

`@torch.no_grad()` 用于关闭梯度记录，适合验证、测试和推理阶段，可以减少显存占用并避免不必要的计算图构建。

</details>
