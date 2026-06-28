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

<details>
<summary>Week 2：融合场景信息的轨迹预测</summary>

## Week 2：融合场景信息的轨迹预测

`week2` 在 Week 1 仅使用历史自车轨迹的基础上，引入自车状态、局部目标点和周围交通参与者信息，使用多分支 MLP 预测未来二维轨迹。同时，训练目标从单一轨迹回归扩展为轨迹、终点、平滑性和碰撞约束组成的联合 loss，用于理解场景信息融合与多目标轨迹学习的基本流程。

### 目录结构与文件目的

```text
week2/
├── train.py
├── eval.py
├── data/
│   ├── dataset.py
│   ├── generate_week2_scene_dataset.py
│   └── week2_synthetic_scene_dataset_v1/
├── models/
│   └── scene_mlp.py
├── losses/
│   └── traj_loss.py
└── utils/
    └── metrics.py
```

`train.py`：训练入口。负责创建数据集和 `DataLoader`，初始化 `SceneMLP`、`TrajLoss` 和 `AdamW` 优化器，执行训练与验证，并在验证集总 loss 变好时保存 `best_model.pt`。

`eval.py`：评估与可视化入口。负责加载训练好的 checkpoint，计算 ADE/FDE，并把若干条样本的历史轨迹、真实未来轨迹和预测未来轨迹画成图片。

`data/generate_week2_scene_dataset.py`：场景感知合成数据生成脚本。生成道路通畅、左右转、跟车、避障和横穿交通等 6 类场景，并保存为 `.npz` 数据文件、`metadata.csv` 元信息和 `schema.json` 数据说明。

`data/dataset.py`：PyTorch 数据集封装。`MLPDataset` 从指定 split 的 `samples.npz` 中读取自车历史轨迹、自车状态、局部目标、周围交通参与者、有效性 mask、未来轨迹和场景类别，并返回模型训练所需的张量字典。

`data/week2_synthetic_scene_dataset_v1/`：已经生成好的示例数据集。包含 `train` 和 `test` 两个 split，每个 split 下有 `samples.npz` 和 `metadata.csv`。

`models/scene_mlp.py`：模型定义。`SceneMLP` 分别编码自车历史轨迹、自车状态、局部目标和周围交通参与者，将各分支特征拼接后，通过轨迹预测头输出 `[batch_size, future_len, 2]` 的未来轨迹。

`losses/traj_loss.py`：联合轨迹预测损失。`TrajLoss` 包含带时间权重的轨迹回归 loss、终点 loss、平滑性 loss 和碰撞 loss；默认权重分别为 `1.0`、`0.2`、`0.05` 和 `0.1`。

`utils/metrics.py`：评估指标。`ade` 计算所有未来时间点的平均欧氏距离误差，`fde` 计算最后一个未来点的欧氏距离误差。

### 数据格式

默认数据集路径：

```text
week2/data/week2_synthetic_scene_dataset_v1
```

数据使用自车中心坐标系，当前自车位置为 `[0, 0]`，`+x` 方向大致指向自车前方，位置单位为米。默认采样间隔为 `0.1 s`，每个样本包含：

```text
ego_history:  [10, 2]，自车历史位置 [x, y]
ego_state:    [4]，自车状态 [速度, 航向角, 加速度, 曲率]
goal:         [2]，局部目标点 [x, y]
agents:       [8, 5]，周围交通参与者 [x, y, vx, vy, radius]
agent_mask:   [8]，有效交通参与者为 1，补齐位置为 0
future:       [20, 2]，自车未来轨迹 [x, y]
scenario_type: 场景类型编号
```

`scenario_type` 与场景名称的对应关系：

```text
0: straight_clear
1: left_turn
2: right_turn
3: follow_vehicle
4: obstacle_avoidance
5: cross_traffic
```

### 使用方法

以下命令默认从 `week2` 目录执行：

```bash
cd Dive-into-e2e-auto-driving/week2
```

如需重新生成合成数据集：

```bash
python data/generate_week2_scene_dataset.py \
  --out data/week2_synthetic_scene_dataset_v1 \
  --train-samples 12000 \
  --test-samples 2400 \
  --history-len 10 \
  --future-len 20 \
  --max-agents 8 \
  --dt 0.1
```

训练模型：

```bash
python train.py \
  --data-root data/week2_synthetic_scene_dataset_v1 \
  --batch-size 256 \
  --epochs 100 \
  --lr 1e-3 \
  --save_dir outputs/checkpoints
```

训练过程中会打印每个 epoch 的训练集和验证集 loss：

```text
loss, traj_loss, goal_loss, smooth_loss, collision_loss
```

总 loss 的计算方式为：

```text
loss = traj_loss
     + 0.2 * goal_loss
     + 0.05 * smooth_loss
     + 0.1 * collision_loss
```

当验证集总 loss 刷新最好结果时，脚本会保存：

```text
outputs/checkpoints/best_model.pt
```

checkpoint 中包含：

```text
epoch: 保存时的 epoch
model_state_dict: 模型参数
optimizer_state_dict: 优化器状态
best_val_traj: 当前最好的验证集总 loss
args: 本次训练使用的命令行参数
```

评估并可视化预测结果：

```bash
python eval.py \
  --data-root data/week2_synthetic_scene_dataset_v1 \
  --checkpoint outputs/checkpoints/best_model.pt \
  --save-dir outputs/figures \
  --batch-size 256 \
  --num-vis 20
```

评估脚本会输出 ADE/FDE，并在 `outputs/figures` 下保存若干张轨迹对比图。图中：

```text
gray: 历史轨迹
green: 真实未来轨迹
red: 模型预测未来轨迹
black: 当前时刻位置
```

### 关键概念

`agent_mask` 用于区分真实交通参与者和定长数组中的补齐位置。编码前将 `agents` 与 mask 相乘，可以避免补齐数据影响场景特征。

多分支特征融合是指先分别编码不同模态的输入，再拼接成统一的场景特征。当前模型分别处理历史轨迹、自车状态、目标点和周围交通参与者。

带时间权重的轨迹 loss 会让较远未来的轨迹点获得更大的权重，使模型更加关注长期预测误差。

终点 loss 约束预测轨迹的最后一点接近局部目标；平滑性 loss 约束相邻速度变化；碰撞 loss 对预测轨迹与有效交通参与者之间小于安全距离的情况进行惩罚。

</details>
