import numpy as np
from sedo.optimizer import SEDOptimizer

# 模拟公交站点之间的距离矩阵
station_distances = np.array([
    [0, 3, 5, 7, 9],
    [3, 0, 2, 4, 6],
    [5, 2, 0, 1, 3],
    [7, 4, 1, 0, 2],
    [9, 6, 3, 2, 0]
])

def objective_function(route_indices):
    """route_indices 是一个排列，如 [0, 2, 1, 4, 3]"""
    route = np.round(np.clip(route_indices, 0, len(station_distances) - 1)).astype(int)
    # 去重并重新排列（关键步骤）
    _, idx = np.unique(route, return_index=True)
    route = route[np.sort(idx)]
    while len(route) < len(station_distances):
        missing = np.setdiff1d(np.arange(len(station_distances)), route)
        route = np.append(route, np.random.choice(missing))
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += station_distances[route[i], route[i+1]]
    return total_distance

opt = SEDOptimizer(
    objective_func=objective_function,
    problem_dim=5,
    n_particles=20,
    bounds=[(0, 4)] * 5,
    discrete_dims=list(range(5)),
    init_method='lhs',
    is_permutation=True
)

opt.optimize(max_iter=100)

best_route = opt.get_best_solution().astype(int)
print("最优公交调度路线:", best_route)