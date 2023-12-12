import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from Optimizers import * 
from utils import *

# 固定随机种子
seed_everything()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义非凸函数
def h(x, y):
    return x**2 - y**2

# 创建三维网格
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = h(X, Y)

# 设置迭代次数
num_iterations = 50

# 初始化参数和优化器
def create_optimizer(opt_name, initial_point, lr=0.01):
    param = torch.tensor(initial_point, requires_grad=True, device=device)
    if opt_name == 'SGD':
        optimizer = SGDOptimizer([param], lr=lr)
    elif opt_name == 'MomentumSGD':
        optimizer = MomentumSGDOptimizer([param], lr=lr, momentum=0.9)
    elif opt_name == 'AdaGrad':
        optimizer = AdagradOptimizer([param], lr=lr)
    elif opt_name == 'RMSProp':
        optimizer = RMSPropOptimizer([param], lr=lr)
    elif opt_name == 'Adam':
        optimizer = AdamOptimizer([param], lr=lr)
    return param, optimizer

# 定义优化器和颜色
opt_names = ['SGD', 'MomentumSGD', 'AdaGrad', 'RMSProp', 'Adam']
colors = ['red', 'green', 'blue', 'purple', 'orange']
initial_point = [0.1, 0.1]  # 避开鞍点(0,0)
params = {}
optimizers = {}
for name, color in zip(opt_names, colors):
    params[name], optimizers[name] = create_optimizer(name, initial_point)

# 为每个优化器创建独立的路径
paths = {name: [] for name in opt_names}

# 执行优化并记录每步路径
for opt_name, optimizer in optimizers.items():
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = h(params[opt_name][0], params[opt_name][1])
        loss.backward()
        optimizer.step()
        paths[opt_name].append(params[opt_name].detach().clone().cpu().numpy())  # Move to CPU for plotting
        if (i + 1) % 10 == 0:
            print(f"Optimizer: {opt_name}, Iteration: {i+1}/{num_iterations}, Loss: {loss.item()}")

# 创建图和轴
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Function Value')
ax.legend()

# 绘制迭代点和路径的函数
lines = {opt_name: ax.plot([], [], [], label=opt_name, color=color)[0] for opt_name, color in zip(opt_names, colors)}
points = {opt_name: ax.scatter([], [], [], color=color, depthshade=False) for opt_name, color in zip(opt_names, colors)}

# 初始化动画函数
def init():
    # 设置初始视图角度
    ax.view_init(elev=30, azim=30)  # elev 和 azim 可以根据需要调整
    return lines.values(), points.values()

# 更新动画的函数
def update(num):
    # 在每次更新前，重新设置视图角度以保持视图不变
    ax.view_init(elev=30, azim=30)  # 保持与init中相同的视图角度
    for opt_name in opt_names:
        line = lines[opt_name]
        point = points[opt_name]
        path_data = np.array(paths[opt_name][:num+1])
        if num > 0 and len(path_data) > 0:
            xdata, ydata = path_data[:, 0], path_data[:, 1]
            zdata = h(torch.tensor(xdata, dtype=torch.float32), torch.tensor(ydata, dtype=torch.float32))
            line.set_data(xdata, ydata)
            line.set_3d_properties(zdata)
            point._offsets3d = (np.array([xdata[-1]]), np.array([ydata[-1]]), np.array([zdata[-1]]))
    ax.legend()
    return [line for line in lines.values()] + [point for point in points.values()]

# 创建动画
ani = FuncAnimation(fig, update, frames=num_iterations, init_func=init, blit=False, repeat=False)

# 保存动画
gif_path = 'Result/Non_convex/optimization_h.gif'
ani.save(gif_path, writer='pillow', fps=10)
print(f"The optimization paths GIF has been saved at '{gif_path}'.")