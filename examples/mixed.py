import numpy as np
from sedo.optimizer import SEDOptimizer

print("=== 科研场景优化 ===")

print("\n1. 公交调度优化")
# 模拟公交站点之间的距离矩阵
station_distances = np.array([
    [0, 3, 5, 7, 9],
    [3, 0, 2, 4, 6],
    [5, 2, 0, 1, 3],
    [7, 4, 1, 0, 2],
    [9, 6, 3, 2, 0]
])

def route_objective(route_indices):
    """route_indices 是一个排列，如 [0, 2, 1, 4, 3]"""
    route = np.round(np.clip(route_indices, 0, len(station_distances) - 1)).astype(int)
    # 去重并重新排列
    _, idx = np.unique(route, return_index=True)
    route = route[np.sort(idx)]
    while len(route) < len(station_distances):
        missing = np.setdiff1d(np.arange(len(station_distances)), route)
        route = np.append(route, np.random.choice(missing))
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += station_distances[route[i], route[i+1]]
    return total_distance

route_opt = SEDOptimizer(
    objective_func=route_objective,
    problem_dim=5,
    n_particles=20,
    bounds=[(0, 4)] * 5,
    discrete_dims=list(range(5)),
    init_method='lhs',
    is_permutation=True
)

route_opt.optimize(max_iter=100)

best_route = route_opt.get_best_solution().astype(int)
print("最优公交调度路线:", best_route)

print("\n2. 多目标旅行商问题")
# 距离矩阵（模拟城市间距离）
distance_matrix = np.array([
    [0, 3, 5, 7, 9],
    [3, 0, 2, 4, 6],
    [5, 2, 0, 1, 3],
    [7, 4, 1, 0, 2],
    [9, 6, 3, 2, 0]
])

# 第二个目标：每个城市的"成本"（例如拥堵指数）
cost_vector = np.array([1, 2, 1, 3, 2])

def multi_objective_tsp(route):
    total_distance = 0
    total_cost = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i+1]]
        total_cost += cost_vector[route[i]] + cost_vector[route[i+1]]
    return [total_distance, total_cost]

multi_opt = SEDOptimizer(
    objective_func=multi_objective_tsp,
    problem_dim=5,
    n_particles=30,
    bounds=[(0, 4)] * 5,
    discrete_dims=list(range(5)),
    is_permutation=True,
    multi_objective=True
)

multi_opt.optimize(max_iter=100)

pareto_front = multi_opt.get_best_solution()
print("Pareto Front:")
for sol in pareto_front:
    print(sol)

print("\n3. 超参数优化")
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

hyper_opt = SEDOptimizer(
    objective_func=evaluate_model,
    problem_dim=3,
    n_particles=20,
    bounds=bounds,
    discrete_dims=[2]  # 批大小是整数
)

hyper_opt.optimize(max_iter=100)

best_params = hyper_opt.get_best_solution()
print("最优超参数:", best_params)


print("\n=== 生活场景优化 ===")

print("\n1. 家电节能调度")
# 定义目标函数：计算总电费（假设每小时电价不同）
def electricity_cost(schedule: np.ndarray) -> float:
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

energy_opt = SEDOptimizer(
    objective_func=electricity_cost,
    problem_dim=5,
    n_particles=20,
    bounds=[(0, 23)] * 5,
    discrete_dims=discrete_dims
)

energy_opt.optimize(max_iter=100)

best_schedule = energy_opt.get_best_solution()
print("最优家电调度时间:", best_schedule)

print("\n2. 旅行路径规划")
# 城市坐标
cities = np.array([
    [0, 0], [1, 2], [3, 1], [5, 3], [2, 5],
    [4, 4], [6, 2], [7, 5], [8, 3], [9, 1]
])

def path_objective(path_indices):
    indices = np.round(np.clip(path_indices, 0, len(cities)-1)).astype(int)
    path = cities[indices]
    distance = 0
    for i in range(len(path) - 1):
        distance += np.linalg.norm(path[i+1] - path[i])
    return distance

path_opt = SEDOptimizer(
    objective_func=path_objective,
    problem_dim=len(cities),
    n_particles=30,
    bounds=[(0, len(cities)-1)] * len(cities),
    discrete_dims=list(range(len(cities))),
    is_permutation=True
)

path_opt.optimize(max_iter=150)
best_path = path_opt.get_best_solution().astype(int)
print("最优城市访问顺序:", best_path)
print("总路径长度:", path_objective(best_path))

print("\n3. 货架布局优化")
# 商品热销指数（越高越受欢迎）
product_popularity = [5, 8, 6, 7, 9, 4, 3, 2]

# 商品相关性矩阵（对称矩阵，值越大表示越常一起购买）
product_correlation = np.random.rand(8, 8)
np.fill_diagonal(product_correlation, 0)

# 货架间距离矩阵（模拟空间布局）
shelf_distances = np.random.randint(1, 10, size=(8, 8))
np.fill_diagonal(shelf_distances, 0)

def layout_objective(layout):
    """
    layout: 排列形式，如 [3, 1, 0, 2, 5, 7, 4, 6]
    返回综合成本：路径长度 + 热门商品距离入口 + 商品关联度惩罚
    """
    layout = np.round(np.clip(layout, 0, 7)).astype(int)
    _, idx = np.unique(layout, return_index=True)
    layout = layout[np.sort(idx)]
    while len(layout) < 8:
        missing = np.setdiff1d(np.arange(8), layout)
        layout = np.append(layout, np.random.choice(missing))

    total_path_length = 0
    for i in range(len(layout) - 1):
        total_path_length += shelf_distances[layout[i], layout[i+1]]

    hot_penalty = sum(product_popularity[p] * (i+1) for i, p in enumerate(layout[:3]))

    correlation_bonus = 0
    for i in range(len(layout) - 1):
        correlation_bonus += product_correlation[layout[i], layout[i+1]]

    return total_path_length + hot_penalty - correlation_bonus

layout_opt = SEDOptimizer(
    objective_func=layout_objective,
    problem_dim=8,
    n_particles=30,
    bounds=[(0, 7)] * 8,
    discrete_dims=list(range(8)),
    init_method='lhs',
    is_permutation=True
)

layout_opt.optimize(max_iter=150)

best_layout = layout_opt.get_best_solution().astype(int)
print("最优货架布局顺序:", best_layout)

print("\n=== 综合结果 ===")
print("最优公交调度路线:", best_route)
print("多目标旅行 Pareto Front:")
for sol in pareto_front:
    print(sol)
print("最优超参数:", best_params)
print("最优家电调度时间:", best_schedule)
print("最优城市访问顺序:", best_path)
print("总路径长度:", path_objective(best_path))
print("最优货架布局顺序:", best_layout)