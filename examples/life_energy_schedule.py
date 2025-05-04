import numpy as np
from sedo.optimizer import SEDOptimizer

# 定义目标函数：计算总电费（假设每小时电价不同）
def objective_function(schedule: np.ndarray) -> float:
    # schedule 是一个长度为 N 的数组，表示每个设备的启动时间（0~23）
    electricity_prices = [0.5, 0.4, 0.3, 0.2, 0.2, 0.3,  # 低谷电价
                          0.6, 0.7, 0.8, 1.0, 1.2, 1.1,  # 高峰电价
                          0.9, 0.8, 0.7, 0.6, 0.5, 0.4,  # 中间段
                          0.3, 0.2, 0.1, 0.2, 0.3, 0.4]  # 深夜
    total_cost = 0
    for t in schedule:
        hour = int(t)
        total_cost += electricity_prices[hour % 24]
    return total_cost

# 离散维度设置：家电只能在整点开启
discrete_dims = list(range(5))  # 假设有5个家电

opt = SEDOptimizer(
    objective_func=objective_function,
    problem_dim=5,
    n_particles=20,
    bounds=[(0, 23)] * 5,
    discrete_dims=discrete_dims
)

opt.optimize(max_iter=100)

best_schedule = opt.get_best_solution()
print("最优家电调度时间:", best_schedule)