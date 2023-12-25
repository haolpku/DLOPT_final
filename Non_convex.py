import torch
from utils import *
from scipy.optimize import fsolve

# 定义非凸函数（双曲抛物面）
def h(x, y):
    return x**2 - y**2

# 定义一个有极小点的非凸函数
def g(x, y):
    return (x**2 - 10) ** 2 + (y**2 - 10) ** 2 + 20*x + 20*y
# 求解函数g的极小值点
def equations(vars):
    x, y = vars
    eq1 = 4*x*(x**2 - 10) + 20
    eq2 = 4*y*(y**2 - 10) + 20
    return [eq1, eq2]
def solve_minimum_g():
    # 解析求解梯度为0的点
    solutions = [fsolve(equations, (guess_x, guess_y)) for 
                 guess_x, guess_y in [(-5, -5), (-5, 5), (5, -5), (5, 5)]]
    # 验证哪个是全局极小值点
    min_value = float('inf')
    min_point = None
    for x, y in solutions:
        value = g(x, y)
        if value < min_value:
            min_value = value
            min_point = (x, y)
    return min_point

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
    losses - 损失在优化过程中的历史值
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
    losses = []
    history.append([p.detach().clone().cpu().numpy() for p in params])
    losses.append(optimize_fn(params[0][0], params[0][1]).item())
    for i in range(num_iterations):
        optimizer.zero_grad()
        # 展开 params 并传递给优化函数
        loss = optimize_fn(params[0][0], params[0][1])
        loss.backward()
        optimizer.step()
        history.append([p.detach().clone().cpu().numpy() for p in params])
        losses.append(loss.item())
        if (i + 1) % 10 == 0:
            print(f"Optimizer: {optimizer_name}, Iteration: {i+1}/{num_iterations}, Loss: {loss.item()}")
    return history, losses

if __name__ == '__main__':
    '''
    print('求解函数g的极小值点：', solve_minimum_g())
    exit()
    '''   
    # 固定随机种子
    seed_everything()

    # 初始设置
    optimizers = ['SGD', 'MomentumSGD', 'AdaGrad', 'RMSProp', 'Adam']

    ## 优化没有极小值点的函数h(x)
    # lr = 0.01
    lr = 0.01
    init_point = [-0.1, 0.001]
    num_steps = 50
    histories = {}
    losses = {}
    for optimizer_name in optimizers:
        histories[optimizer_name], losses[optimizer_name] = optimize_function_with_optimizers(h, optimizer_name, lr, num_steps, init_point)
    plot_optimization_comparison(histories, h, plot_type='surface', save_gif=True, gif_path='Result/Non_convex/optimization_h_sf_lr001.gif')
    # lr = 0.1
    lr = 0.1
    histories = {}
    losses = {}
    seed_everything()
    for optimizer_name in optimizers:
        histories[optimizer_name], losses[optimizer_name] = optimize_function_with_optimizers(h, optimizer_name, lr, num_steps, init_point)
    plot_optimization_comparison(histories, h, plot_type='surface', save_gif=True, gif_path='Result/Non_convex/optimization_h_sf_lr01.gif')

    ## 优化非凸有极小值的函数g(x)
    init_point = [0., -2.]
    num_steps = 500
    minimal = g(-3.3876190584754156, -3.3876190584754156)

    # lr = 0.01
    lr = 0.01
    histories = {}
    losses = {}
    seed_everything()
    for optimizer_name in optimizers:
        histories[optimizer_name], losses[optimizer_name] = optimize_function_with_optimizers(g, optimizer_name, lr, num_steps, init_point)
    plot_function_value(losses, minimal, optimizers, 'Result/Non_convex')
    plot_optimization_comparison(histories, g, plot_type='contour', save_gif=True, gif_path='Result/Non_convex/optimization_g_ct_lr001.gif')

    # AdaGrad的lr = 0.05，其余lr = 0.01
    histories = {}
    losses = {}
    lr_AdaGrad = 0.05
    seed_everything()
    for optimizer_name in optimizers:
        if optimizer_name == 'AdaGrad':
            histories[optimizer_name], losses[optimizer_name] = optimize_function_with_optimizers(g, optimizer_name, lr_AdaGrad, num_steps, init_point)
        else:
            histories[optimizer_name], losses[optimizer_name] = optimize_function_with_optimizers(g, optimizer_name, lr, num_steps, init_point)
    plot_function_value(losses, minimal, optimizers, 'Result/Non_convex')
    plot_optimization_comparison(histories, g, plot_type='contour', save_gif=True, gif_path='Result/Non_convex/optimization_g_ct_lr001_005.gif')

    # AdaGrad的lr = 0.15，其余lr = 0.01
    histories = {}
    losses = {}
    lr_AdaGrad = 0.1
    seed_everything()
    for optimizer_name in optimizers:
        if optimizer_name == 'AdaGrad':
            histories[optimizer_name], losses[optimizer_name] = optimize_function_with_optimizers(g, optimizer_name, lr_AdaGrad, num_steps, init_point)
        else:
            histories[optimizer_name], losses[optimizer_name] = optimize_function_with_optimizers(g, optimizer_name, lr, num_steps, init_point)
    plot_function_value(losses, minimal, optimizers, 'Result/Non_convex')
    plot_optimization_comparison(histories, g, plot_type='contour', save_gif=True, gif_path='Result/Non_convex/optimization_g_ct_lr001_015.gif')
