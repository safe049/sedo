# 示例：多目标旅行商问题
import numpy as np
from sedo.optimizer import SEDOptimizer

# 距离矩阵（模拟城市间距离）
distance_matrix = np.array([
    [0, 3, 5, 7, 9],
    [3, 0, 2, 4, 6],
    [5, 2, 0, 1, 3],
    [7, 4, 1, 0, 2],
    [9, 6, 3, 2, 0]
])

# 第二个目标：每个城市的“成本”（例如拥堵指数）
cost_vector = np.array([1, 2, 1, 3, 2])

def multi_objective_tsp(route):
    total_distance = 0
    total_cost = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i+1]]
        total_cost += cost_vector[route[i]] + cost_vector[route[i+1]]
    return [total_distance, total_cost]

opt = SEDOptimizer(
    objective_func=multi_objective_tsp,
    problem_dim=5,
    n_particles=30,
    bounds=[(0, 4)] * 5,
    discrete_dims=list(range(5)),
    is_permutation=True,
    multi_objective=True
)

opt.optimize(max_iter=100)

pareto_front = opt.get_best_solution()
print("Pareto Front:")
for sol in pareto_front:
    print(sol)