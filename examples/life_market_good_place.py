import numpy as np
from sedo.optimizer import SEDOptimizer

# 商品热销指数（越高越受欢迎）
product_popularity = [5, 8, 6, 7, 9, 4, 3, 2]

# 商品相关性矩阵（对称矩阵，值越大表示越常一起购买）
product_correlation = np.random.rand(8, 8)
np.fill_diagonal(product_correlation, 0)

# 货架间距离矩阵（模拟空间布局）
shelf_distances = np.random.randint(1, 10, size=(8, 8))
np.fill_diagonal(shelf_distances, 0)

def objective_function(layout):
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

opt = SEDOptimizer(
    objective_func=objective_function,
    problem_dim=8,
    n_particles=30,
    bounds=[(0, 7)] * 8,
    discrete_dims=list(range(8)),
    init_method='lhs',
    is_permutation=True
)

opt.optimize(max_iter=150)

best_layout = opt.get_best_solution().astype(int)
print("最优货架布局顺序:", best_layout)