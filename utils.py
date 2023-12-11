import matplotlib.pyplot as plt 
import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    """
    固定所有可能影响结果可复现性的随机数种子。
    
    :param seed: 随机数种子，默认为 42。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_performance(losses, accuracies, optimizers, save_path):
    """
    分别绘制每种优化器的损失和准确度曲线，并将图表保存到指定路径。

    :param losses: 损失字典，键为优化器名称，值为损失列表。
    :param accuracies: 准确度字典，键为优化器名称，值为准确度列表。
    :param optimizers: 优化器名称列表。
    :param save_path: 保存图表的路径。
    """
    # 设置图表样式
    markers = ['o', 's', '^', 'D', '*']  # 不同的标记符号
    colors = ['b', 'g', 'r', 'c', 'm']   # 不同的颜色

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    for i, opt_name in enumerate(optimizers):
        plt.plot(losses[opt_name], marker=markers[i], color=colors[i], linestyle='-', label=f'{opt_name} Loss')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training Loss Comparison', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_comparison.png'))
    plt.close()

    # 绘制准确度曲线
    plt.figure(figsize=(10, 5))
    for i, opt_name in enumerate(optimizers):
        plt.plot(accuracies[opt_name], marker=markers[i], color=colors[i], linestyle='--', label=f'{opt_name} Accuracy')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Test Accuracy Comparison', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'accuracy_comparison.png'))
    plt.close()


