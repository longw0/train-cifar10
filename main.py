import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from torchinfo import summary
from torchviz import make_dot

"""
CIFAR-10 官方训练集 50000 张，测试集 10000 张。
"""


def load_img():
    # 这段 train_transform 做了 6 件事：随机裁剪、随机翻转、颜色抖动、转张量、归一化、随机擦除。前三个和最后一个是为了数据增强、抑制过拟合；
    # 中间两个是 PyTorch 训练必需的数据格式转换。验证/测试集只做 ToTensor + Normalize，不做随机增强。
    train_transform = transforms.Compose([
        # 先在图像四周各填充 4 像素（32×32 → 40×40），然后随机裁剪出 32×32 的区域。效果：每次训练看到的图，位置略有偏移。让模型不依赖物体的绝对位置，提升平移鲁棒性。
        transforms.RandomCrop(32, padding=4),
        # 以 50% 概率把图左右翻转，增加样本多样性。
        transforms.RandomHorizontalFlip(p=0.5),
        # 亮度、对比度、饱和度随机变化 ±20%。让模型适应不同光照、颜色条件，不依赖特定颜色。
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # 把 PIL 图像（0~255 整数）转成 PyTorch 张量（0~1 浮点）。CIFAR-10 图从 (32, 32, 3) 变成 (3, 32, 32)。作用：PyTorch 模型要求张量输入。
        transforms.ToTensor(),
        # 对每个通道做：(x - mean) / std
        # 原始图像像素在 0~255，ToTensor 后缩到 0~1。但这个分布不居中、尺度不统一，直接训练容易收敛慢、梯度消失。
        # 归一化按每个通道的均值和标准差做标准化，让数据变成均值 0、标准差 1 的分布，大部分值落在 -2 到 +2 之间。
        # 这样梯度更健康，训练更稳定、更快。
        # 就像现在，原始数据的额均值是[0.4914, 0.4822, 0.4465]，标准差是[0.2470, 0.2435, 0.2616]，归一化以后，均值就是0，标准差是1
        # 里面的mean和std是统计数据集里的所有RGB通道，然后算出来的
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        # 以 25% 概率，在图上随机选一块矩形区域，用随机值或 0 填充。作用：强迫模型不依赖局部特征，学会从全局判断，抑制过拟合。
        transforms.RandomErasing(p=0.25)
    ])

    # 验证/测试：只做归一化，不做增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    # CIFAR-10：train=True 是 50000 张训练集，train=False 是 10000 张测试集
    full_train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # 从训练集划验证集。验证集要用 test_transform，不能带增强。
    # 从训练集里划 10% 做验证集。
    val_size = int(len(full_train_set) * 0.1)
    train_size = len(full_train_set) - val_size
    train_subset, val_subset = random_split(full_train_set, [train_size, val_size],generator=torch.Generator().manual_seed(42))
    # 覆盖验证集的 transform
    val_subset.dataset = copy.deepcopy(full_train_set)
    val_subset.dataset.transform = test_transform

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader


def create_model(num_classes=10):
    """
    从零训练的 ResNet-18，针对 CIFAR-10 的 32x32 输入做适配：
    - 首层 7x7 stride=2 改成 3x3 stride=1，保留更多空间信息
    - 去掉 maxpool，避免过早下采样
    - 分类头前加 Dropout，抑制过拟合
    """
    model = models.resnet18(weights=None)

    # 适配 32*32，若是按照原来的 7×7, stride=2, padding=3 进行输入，经过conv1以后就变成16*16，
    # 再经过 maxpool 以后就比那成 8*8，到 layer4 就会变成 1*1，空间信息就没了
    # 修改完以后，经过 conv1 和 maxpool，尺寸还是 32*32，这样的话，到 layer4 就会变成 4*4，保留了从简信息
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # 分类头 + Dropout
    # 因为 ResNet18 默认的是 1000 个分类，所以要适配 CIFAR10 的 10 个分类，所以要用nn.Linear(in_features, num_classes)
    # 因为原来ResNet18用的是125w的数据去训练，所以不太会过拟合，但是 CIFAR10 的训练数据只有4.5w，所以要手动加Dropout
    # 在分类头前随机丢弃一半神经元，强迫模型不依赖某几个特定神经元，提升泛化能力。
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    return total_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    return total_loss / total, 100.0 * correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    # 超参数
    num_epochs = 1

    train_loader, val_loader, test_loader = load_img()
    model = create_model(num_classes=10).to(device)

    # Loss = -log(p_真实类概率) ，真实类的概率越高，Loss 越小；给得越低，Loss 越大。
    criterion = nn.CrossEntropyLoss()
    # 随机梯度下降：每次只用一批样本，不是全部，有随机性。
    # momentum：保留上一次更新方向的一部分，像小球滚下山有惯性。
    # weight_decay：L2 正则，让参数别太大，抑制过拟合。
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # SGD 对学习率敏感，所以需要手动调学习率，但配合 CosineAnnealing 后，最终准确率往往更高。
    # CosineAnnealing 按余弦曲线让 LR 从 0.1 降到接近 0。
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc, best_epoch, counter, patience = 0.0, 0, 0, 15
    save_path = 'result/best_resnet18.pth'

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.5f}")

        # 保存最佳模型 + 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            counter = 0
            # torch.save(model.state_dict(), save_path)
            # print(f"  -> Best model saved (val_acc={best_val_acc:.2f}%)")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}, Best val acc: {best_val_acc:.2f}%")
                break

    # 5. 用测试集评估最优模型
    # model.load_state_dict(torch.load(save_path, map_location=device))
    # test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    # print(f"\nFinal Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # 6. 画曲线
    # draw_plot(history)


def draw_plot(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='val')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('curve.png')
    plt.show()
    print("Curve saved to curve.png")


def show_summary():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes=10).to(device)
    summary(model, input_size=(1, 3, 32, 32))
    x = torch.randn(1, 3, 32, 32).to(device)
    make_dot(model(x), params=dict(model.named_parameters())).render("model_structure", format="png")
    torch.onnx.export(model, x, "./result/model.onnx", input_names=["input"], output_names=["output"])


if __name__ == '__main__':
    # show_summary()
    main()
