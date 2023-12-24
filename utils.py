import matplotlib.pyplot as plt 
import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Optimizers import *

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
    plt.savefig(os.path.join(save_path, 'loss_comparison_2.png'))
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
    plt.savefig(os.path.join(save_path, 'accuracy_comparison_2.png'))
    plt.close()

def plot_optimization_comparison(histories, optimize_fn, plot_type='contour', save_gif=True, gif_path='Result/Non_convex/optimization.gif'):
    '''
    绘制不同优化器优化过程的比较动画
    
    参数：
    histories - 各优化器的历史数据字典，每个键为优化器名称，值为优化过程中的参数记录
    optimize_fn - 优化的目标函数。
    plot_type - 绘图类型，'contour' 或 'surface'
    save_gif - 是否保GIF，默认True保存
    gif_path - GIF的保存路径

    返回：
    ani - 优化过程的GIF
    '''
    if plot_type == 'surface':
        fig = plt.figure(figsize=(10, 7.5))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=30)
        x = np.linspace(-1.5, 1.5, 200)
        y = np.linspace(-1.5, 1.5, 200)
        X, Y = np.meshgrid(x, y)
        Z = optimize_fn(torch.tensor(X), torch.tensor(Y))
        ax.plot_surface(X, Y, Z.detach().numpy(), rstride=1, cstride=1, cmap='viridis', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Function Value')
        ax.legend()
    elif plot_type == 'contour':
        fig, ax = plt.subplots(figsize=(10, 7.5))
        x = np.linspace(-6, 6, 200)
        y = np.linspace(-6, 6, 200)
        X, Y = np.meshgrid(x, y)
        Z = optimize_fn(torch.tensor(X), torch.tensor(Y))
        contour = ax.contour(X, Y, Z.detach().numpy(), levels=20)
        plt.colorbar(contour)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        minima_x, minima_y = -3.3876190584754156, -3.3876190584754156
        ax.scatter(minima_x, minima_y, color='grey', edgecolors='black', marker='*', s=300, label='Minimum')
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    opt_names = list(histories.keys())
    lines = {}
    points = {}
    for opt_name, color in zip(opt_names, colors):
        if plot_type == 'contour':
            lines[opt_name], = ax.plot([], [], label=opt_name, color=color)
            points[opt_name], = ax.plot([], [], 'o', color=color)
        elif plot_type == 'surface':
            lines[opt_name], = ax.plot([], [], [], label=opt_name, color=color)
            points[opt_name] = ax.scatter([], [], [], color=color, depthshade=False)
        def init():
            for opt_name in opt_names:
                if plot_type == 'contour':
                    lines[opt_name].set_data([], [])
                    points[opt_name].set_data([], [])
                elif plot_type == 'surface':
                    lines[opt_name].set_3d_properties([])
                    points[opt_name]._offsets3d = ([], [], [])
            return [lines[opt_name] for opt_name in opt_names] + [points[opt_name] for opt_name in opt_names]
        def animate(i):
            for opt_name in opt_names:
                full_history = np.array(histories[opt_name])  # 转换为 NumPy 数组
                if i < len(full_history):
                    # 提取前 i 步的历史记录
                    history_slice = full_history[:i+1].reshape(-1, 2) 
                    x_data = history_slice[:, 0]
                    y_data = history_slice[:, 1]
                    if plot_type == 'contour':
                        lines[opt_name].set_data(x_data, y_data)
                        points[opt_name].set_data(full_history[i, 0, 0], full_history[i, 0, 1])
                    elif plot_type == 'surface':
                        z_data = optimize_fn(torch.tensor(x_data), torch.tensor(y_data)).numpy()
                        lines[opt_name].set_data(x_data, y_data)
                        lines[opt_name].set_3d_properties(z_data)
                        points[opt_name]._offsets3d = (np.array([full_history[i, 0, 0]]), 
                                                       np.array([full_history[i, 0, 1]]), 
                                                       np.array([optimize_fn(torch.tensor(full_history[i, 0, 0]), torch.tensor(full_history[i, 0, 1])).numpy()]))
            return [lines[opt_name] for opt_name in opt_names] + [points[opt_name] for opt_name in opt_names]    
    ax.legend()
    ani = FuncAnimation(fig, animate, init_func=init, frames=max(len(h) for h in histories.values()), interval=200, blit=True, repeat=True)
    plt.show()
    # 保存动画
    if save_gif:
        ani.save(gif_path, writer='pillow', fps=10)
        print(f"The optimization paths GIF has been saved at '{gif_path}'.")
    return ani
    
def plot_function_value(histories, optimizers, save_path):
    """
    分别绘制每种优化器的函数值下降曲线，并将图表保存到指定路径。

    :param histories: 函数字典，键为优化器名称，值为函数值列表。
    :param optimizers: 优化器名称列表。
    :param save_path: 保存图表的路径。
    """
    # 设置图表样式
    markers = ['o', 's', '^', 'D', '*']  # 不同的标记符号
    colors = ['b', 'g', 'r', 'c', 'm']   # 不同的颜色

    # 绘制函数值曲线
    plt.figure(figsize=(10, 5))
    for i, opt_name in enumerate(optimizers):
        plt.plot(histories[opt_name], color=colors[i], marker=markers[i], linestyle='-', label=f'{opt_name} Function Value')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Function Value', fontsize=14)
    plt.title('Function Value Comparison', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'fval.pdf'))

