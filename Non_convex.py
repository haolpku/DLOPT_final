import torch
from utils import *

# 定义非凸函数（双曲抛物面）
def h(x, y):
    return x**2 - y**2

# 定义一个有极小点的非凸函数
def g(x, y):
    return (x**2 - 10) ** 2 + (y**2 - 10) ** 2 + 20*x + 20*y

def optimize_function_with_optimizers(optimize_fn, optimizer_name, lr=0.01, num_iterations=50, init_point=[0., 0.]):
    '''
    通用的优化函数，用于优化给定的函数
    
    参数:
    optimize_fn - 要优化的函数
    optimizer_class - 使用的优化器类
    lr - 学习率
    num_iterations - 优化步数
    init_point - 优化的初始点
    
    返回:
    history - 参数在优化过程中的历史值
    '''  
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 选择优化器
    optimizers = {
    'SGD': SGDOptimizer,
    'MomentumSGD': MomentumSGDOptimizer,
    'AdaGrad': AdagradOptimizer,
    'RMSProp': RMSPropOptimizer,
    'Adam': AdamOptimizer
    }    
    optimizer_class = optimizers[optimizer_name]
    # 初始化参数
    params = [torch.tensor(init_point, requires_grad=True, device=device)]
    optimizer = optimizer_class(params, lr=lr)
    # 记录优化过程中的参数值
    history = []
    for i in range(num_iterations):
        optimizer.zero_grad()
        # 展开 params 并传递给优化函数
        loss = optimize_fn(params[0][0], params[0][1])
        loss.backward()
        optimizer.step()
        history.append([p.detach().clone().cpu().numpy() for p in params])
        if (i + 1) % 10 == 0:
            print(f"Optimizer: {optimizer_name}, Iteration: {i+1}/{num_iterations}, Loss: {loss.item()}")
    return history

# 固定随机种子
seed_everything()

# 初始设置
lr = 0.01
optimizers = ['SGD', 'MomentumSGD', 'AdaGrad', 'RMSProp', 'Adam']

## 优化h(x)
init_point = [0.1, 0.1]
num_steps = 50
histories = {}
for optimizer_name in optimizers:
    histories[optimizer_name] = optimize_function_with_optimizers(h, optimizer_name, lr, num_steps, init_point)
plot_optimization_comparison(histories, h, plot_type='surface', save_gif=True, gif_path='Result/Non_convex/optimization_h_lr001.gif')

## 优化非凸有极小值的函数g(x)
init_point = [0., -2.]
num_steps = 500

# lr = 0.01
lr = 0.01
histories = {}
seed_everything()
for optimizer_name in optimizers:
    histories[optimizer_name] = optimize_function_with_optimizers(g, optimizer_name, lr, num_steps, init_point)
plot_optimization_comparison(histories, g, plot_type='contour', save_gif=True, gif_path='Result/Non_convex/optimization_g_lr001.gif')

# AdaGrad的lr = 0.05，其余lr = 0.01
histories = {}
lr_AdaGrad = 0.05
seed_everything()
for optimizer_name in optimizers:
    if optimizer_name == 'AdaGrad':
        histories[optimizer_name] = optimize_function_with_optimizers(g, optimizer_name, lr_AdaGrad, num_steps, init_point)
    else:
        histories[optimizer_name] = optimize_function_with_optimizers(g, optimizer_name, lr, num_steps, init_point)
plot_optimization_comparison(histories, g, plot_type='contour', save_gif=True, gif_path='Result/Non_convex/optimization_g_lr001_005.gif')

# AdaGrad的lr = 0.1，其余lr = 0.01
histories = {}
lr_AdaGrad = 0.1
seed_everything()
for optimizer_name in optimizers:
    if optimizer_name == 'AdaGrad':
        histories[optimizer_name] = optimize_function_with_optimizers(g, optimizer_name, lr_AdaGrad, num_steps, init_point)
    else:
        histories[optimizer_name] = optimize_function_with_optimizers(g, optimizer_name, lr, num_steps, init_point)
plot_optimization_comparison(histories, g, plot_type='contour', save_gif=True, gif_path='Result/Non_convex/optimization_g_lr001_01.gif')
