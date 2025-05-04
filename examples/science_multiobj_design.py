import numpy as np
from sedo.optimizer import SEDOptimizer

# 多目标函数：返回两个目标值
def multi_objective_func(x):
    weight = x[0]**2 + x[1]**2      # 结构重量
    stress = abs(np.sin(x[0])) + abs(np.cos(x[1]))  # 应力
    return [weight, stress]

# 变量范围
bounds = [(-5, 5), (-5, 5)]

opt = SEDOptimizer(
    objective_func=multi_objective_func,
    problem_dim=2,
    n_particles=50,
    bounds=bounds,
    multi_objective=True
)

opt.optimize(max_iter=100)

pareto_front = opt.get_best_solution()
print("Pareto前沿解集:")
for sol in pareto_front:
    print(sol)