# CIFAR-10 ResNet-18 图像分类

基于 PyTorch 实现的 CIFAR-10 图像分类项目，从零训练适配 32×32 输入的 ResNet-18，包含完整的数据增强、训练、验证与测试流程。

## 项目简介

本项目在 CIFAR-10 数据集上从零训练 ResNet-18，针对 32×32 小尺寸输入对标准 ResNet-18 做了结构适配，并加入多种数据增强与正则化手段抑制过拟合。代码结构清晰，适合作为深度学习入门 / 课程作业参考。

### 主要特性

- **模型适配小尺寸输入**：将首层 7×7 stride=2 卷积改为 3×3 stride=1，并去掉 maxpool，避免过早下采样导致空间信息丢失。
- **丰富的数据增强**：随机裁剪、随机水平翻转、颜色抖动、随机擦除（RandomErasing）。
- **标准化预处理**：使用 CIFAR-10 通道均值 / 标准差归一化。
- **训练策略**：SGD + Momentum + Weight Decay，配合 CosineAnnealingLR 学习率调度。
- **过拟合抑制**：分类头前加入 Dropout(p=0.5)，并使用早停（Early Stopping）。
- **可视化工具**：支持绘制训练曲线、打印模型结构摘要、导出 ONNX 及计算图。

## 环境配置

### 创建 conda 环境

```shell
conda create -n cifar10_gpu python=3.10 -y
conda activate cifar10_gpu
```



### 安装 PyTorch（CUDA 版本）

> 开发设备参考：
>
> - CPU: i7-11700 2.5GHz
> - GPU: GTX 1660Ti
> - 内存: 2×16GB DDR4 3200



```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```



如需 CPU 版本，请前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应命令。

### 安装其他依赖

shell

```
pip install matplotlib numpy pillow tqdm torchinfo torchviz graphviz
```



### 验证安装

python

```
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```



## 快速开始

### 1. 准备目录

训练过程中会保存最佳模型到 `result/` 目录，请先创建：

shell

```
mkdir -p result
```



### 2. 运行训练

shell

```
python main.py
```



首次运行会自动下载 CIFAR-10 数据集到 `./data` 目录。

### 3. 查看模型结构与导出

编辑 `main.py` 末尾，将 `main()` 换成 `show_summary()`：

python

```
if __name__ == '__main__':
    show_summary()
    # main()
```



该函数会：

- 用 `torchinfo` 打印模型参数量与每层输出尺寸；
- 用 `torchviz` 生成计算图 `model_structure.png`；
- 将模型导出为 `./result/model.onnx`。

## 项目结构

text

```
.
├── main.py                  # 主程序：数据加载、模型构建、训练与评估
├── README.md                # 项目说明
├── running.log              # 完整训练日志
├── curve.png                # 训练曲线图 (准确率 & 损失)
├── model_structure.png      # 模型计算图 (torchviz)
├── data/                    # CIFAR-10 数据集（运行后自动生成）
└── result/                  # 输出目录
    ├── best_resnet18.pth    # 最佳模型权重
    └── model.onnx           # 导出的 ONNX 模型
```



## 核心实现说明

### 数据集

CIFAR-10 官方训练集 50000 张、测试集 10000 张，共 10 个类别。本项目从训练集中划分 10% 作为验证集：

| 划分   | 数量  |
| :----- | :---- |
| 训练集 | 45000 |
| 验证集 | 5000  |
| 测试集 | 10000 |

### 数据增强（仅训练集）

| 变换                          | 作用                                         |
| :---------------------------- | :------------------------------------------- |
| `RandomCrop(32, padding=4)`   | 随机位置裁剪，提升平移鲁棒性                 |
| `RandomHorizontalFlip(p=0.5)` | 随机水平翻转，增加样本多样性                 |
| `ColorJitter(0.2, 0.2, 0.2)`  | 亮度 / 对比度 / 饱和度扰动，适应不同光照条件 |
| `ToTensor()`                  | PIL 图像转 PyTorch 张量                      |
| `Normalize(...)`              | 按通道均值 / 标准差标准化                    |
| `RandomErasing(p=0.25)`       | 随机擦除局部区域，强迫模型关注全局特征       |

验证集 / 测试集只做 `ToTensor` + `Normalize`，不做随机增强。

### 模型结构适配

标准 ResNet-18 是为 ImageNet 的 224×224 输入设计的。若直接用于 32×32 的 CIFAR-10：

- 7×7 stride=2 卷积 → 16×16
- maxpool → 8×8
- 到 layer4 时只剩 1×1，空间信息几乎丢失

因此本项目做了如下修改：

python

```
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
```



修改后经过 conv1 与 maxpool 尺寸仍为 32×32，到 layer4 时为 4×4，保留了足够空间信息。

分类头替换为：

python

```
model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))
```



### 训练配置

| 超参数                  | 取值              |
| :---------------------- | :---------------- |
| Epochs                  | 150               |
| Batch Size              | 128               |
| Optimizer               | SGD               |
| Learning Rate           | 0.1               |
| Momentum                | 0.9               |
| Weight Decay            | 5e-4              |
| LR Scheduler            | CosineAnnealingLR |
| Loss                    | CrossEntropyLoss  |
| Early Stopping Patience | 15                |

## 训练结果与可视化

完整训练 150 个 epoch 后，验证集最高准确率达到 **96.24%**，最终在 10000 张测试集上取得：

> **Final Test Loss: 0.1580, Test Acc: 95.57%**

### 训练曲线

![](.\result\curve.png)

左图为准确率曲线，右图为损失曲线。可以看到模型在约 120 个 epoch 后趋于收敛，验证集准确率稳定在 95% 以上。

### 模型计算图

![](.\result\model_structure.png)

> 上图为通过 `torchviz` 生成的 ResNet-18（适配 CIFAR-10 版本）计算图，可见主干卷积残差结构与最后的全局平均池化、全连接分类头。

### 训练日志

完整的运行日志见 `result/running.log`，以下摘录关键片段：

```text
Device: cuda
Train: 45000 | Val: 5000 | Test: 10000
Epoch [1/150]   Train Loss: 2.3266, Acc: 24.11% | Val Loss: 1.7190, Acc: 35.86% | LR: 0.09999 -> Best model saved (val_acc=35.86%)
...
Epoch [112/150] Train Loss: 0.1127, Acc: 96.27% | Val Loss: 0.1892, Acc: 94.10% | LR: 0.01502 -> Best model saved (val_acc=94.10%)
...
Epoch [146/150] Train Loss: 0.0257, Acc: 99.24% | Val Loss: 0.1445, Acc: 96.24% | LR: 0.00018 -> Best model saved (val_acc=96.24%)
Epoch [150/150] Train Loss: 0.0267, Acc: 99.16% | Val Loss: 0.1440, Acc: 96.24% | LR: 0.00000

Final Test Loss: 0.1580, Test Acc: 95.57%
Curve saved to curve.png
```



## 输出说明

训练过程中每个 epoch 会打印：

```text
Epoch [1/150] Train Loss: x.xxxx, Acc: xx.xx% | Val Loss: x.xxxx, Acc: xx.xx% | LR: 0.xxxxx
```



训练结束后调用 `draw_plot(history)` 绘制训练 / 验证的准确率与损失曲线，保存为 `curve.png`。

## License

本项目仅用于学习与研究目的。
