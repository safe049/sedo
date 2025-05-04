import numpy as np
from sedo.optimizer import SEDOptimizer

# 城市坐标
cities = np.array([
    [0, 0], [1, 2], [3, 1], [5, 3], [2, 5],
    [4, 4], [6, 2], [7, 5], [8, 3], [9, 1]
])

def objective_function(path_indices):
    indices = np.round(np.clip(path_indices, 0, len(cities)-1)).astype(int)
    path = cities[indices]
    distance = 0
    for i in range(len(path) - 1):
        distance += np.linalg.norm(path[i+1] - path[i])
    return distance

opt = SEDOptimizer(
    objective_func=objective_function,
    problem_dim=len(cities),
    n_particles=30,
    bounds=[(0, len(cities)-1)] * len(cities),
    discrete_dims=list(range(len(cities))),
    is_permutation=True
)

opt.optimize(max_iter=150)
best_path = opt.get_best_solution().astype(int)
print("最优城市访问顺序:", best_path)
print("总路径长度:", objective_function(best_path))