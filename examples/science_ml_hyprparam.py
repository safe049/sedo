import numpy as np
from sedo.optimizer import SEDOptimizer

# 模拟模型评估函数
def evaluate_model(params):
    lr, reg, batch_size = params
    # 模拟损失值，这里只是演示
    loss = (lr - 0.001)**2 + (reg - 0.01)**2 + (batch_size - 64)**2
    return loss

# 参数范围
bounds = [
    (0.0001, 0.1),   # 学习率
    (0.001, 0.1),    # 正则化系数
    (16, 256)        # 批大小
]

opt = SEDOptimizer(
    objective_func=evaluate_model,
    problem_dim=3,
    n_particles=20,
    bounds=bounds,
    discrete_dims=[2]  # 批大小是整数
)

opt.optimize(max_iter=100)

best_params = opt.get_best_solution()
print("最优超参数:", best_params)