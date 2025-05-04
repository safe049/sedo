import numpy as np
from sedo.optimizer import SEDOptimizer
from sedo.search import SEDSearchCV
from sedo.utils import plot_convergence, save_checkpoint, load_checkpoint

# 单目标测试函数: Sphere 函数
def sphere(x):
    return sum(xi**2 for xi in x)

# 多目标测试函数: ZDT1 问题
def zdt1(x):
    n = len(x)
    f1 = x[0]
    
    g = 1 + 9 * sum(x[1:]) / (n - 1)
    h = 1 - np.sqrt(f1 / g)
    
    f2 = g * h
    return [f1, f2]

# 测试SED优化器的全部功能
def test_sedo_all_features():
    print("=== 开始单目标优化测试 ===")
    
    # Sphere函数优化
    optimizer = SEDOptimizer(
        objective_func=sphere,
        problem_dim=10,
        n_particles=30,
        bounds=[(-5, 5)] * 10,
        use_parallel=True,
        init_method='lhs',
        barrier_height=0.5,
        entropy_threshold=0.8
    )
    
    print("\n单目标优化过程:")
    print("- 使用并行计算")
    print("- 使用LHS初始化方法")
    print("- 粒子数量: 30")
    print("- 问题维度: 10")
    print("- 搜索范围: [-5, 5] × 10")
    print("- 势垒高度: 0.5")
    print("- 熵阈值: 0.8")
    
    optimizer.optimize(max_iter=100)
    best_solution = optimizer.get_best_solution()
    print(f"\nSphere函数最优解: {best_solution}")
    print(f"最优值: {optimizer.global_best_fit}")
    
    # 保存和加载检查点
    save_checkpoint(optimizer, "sedo_checkpoint.pkl")
    loaded_optimizer = load_checkpoint("sedo_checkpoint.pkl")
    print("\n从检查点恢复的最优值:", loaded_optimizer['global_best_fit'])
    
    # 绘制收敛曲线
    plot_convergence(optimizer.history)
    
    print("\n=== 开始多目标优化测试 ===")
    
    # ZDT1多目标优化
    multi_optimizer = SEDOptimizer(
        objective_func=zdt1,
        problem_dim=10,
        n_particles=50,
        bounds=[(0, 1)] * 10,
        multi_objective=True,
        use_parallel=True
    )
    
    print("\n多目标优化过程:")
    print("- 使用并行计算")
    print("- 粒子数量: 50")
    print("- 问题维度: 10")
    print("- 搜索范围: [0, 1] × 10")
    
    multi_optimizer.optimize(max_iter=200)
    pareto_front = multi_optimizer.get_best_solution()
    print(f"\nZDT1问题找到的Pareto前沿包含 {len(pareto_front)} 个解")
    
    # 绘制多目标收敛曲线
    plot_convergence(multi_optimizer.history)
    
    print("\n=== 开始混合变量优化测试 ===")
    
    # 混合连续和离散变量优化
    def mixed_function(x):
        # 前两个变量为整数，后三个为连续变量
        return abs(x[0] - 2) + abs(x[1] - 5) + sum((xi - 1)**2 for xi in x[2:])
    
    mixed_optimizer = SEDOptimizer(
        objective_func=mixed_function,
        problem_dim=5,
        n_particles=20,
        bounds=[(0, 5), (3, 7)] + [(0, 2)] * 3,  # 前两个维度为离散变量
        discrete_dims=[0, 1],
        use_parallel=False
    )
    
    print("\n混合变量优化过程:")
    print("- 不使用并行计算")
    print("- 粒子数量: 20")
    print("- 问题维度: 5")
    print("- 离散变量: 前2个维度")
    print("- 搜索范围: [0, 5], [3, 7], [0, 2] × 3")
    
    mixed_optimizer.optimize(max_iter=150)
    best_mixed = mixed_optimizer.get_best_solution()
    print(f"\n混合变量函数最优解: {best_mixed}")
    print(f"最优值: {mixed_optimizer.global_best_fit}")
    
    # 使用不同的初始化方法
    print("\n使用不同初始化方法进行优化:")
    for init_method in ['uniform', 'lhs', 'orthogonal']:
        print(f"\n使用 {init_method} 初始化方法:")
        test_optimizer = SEDOptimizer(
            objective_func=sphere,
            problem_dim=5,
            n_particles=15,
            bounds=[(-3, 3)] * 5,
            init_method=init_method
        )
        test_optimizer.optimize(max_iter=50)
        print(f"最终最优值: {test_optimizer.global_best_fit}")

def dummy_model(params):
    # 模拟一个带有参数的模型
    x0, x1, x2 = params
    return (x0 - 1) ** 2 + (x1 - 2) ** 2 + (x2 - 3) ** 2

# 测试SEDSearchCV的功能
def test_sedsearchcv():
    print("\n=== 开始SEDSearchCV测试 ===")
    
    # 定义一个简单的模型评估函数

    
    # 参数空间定义
    param_space = {
        'x0': [0, 2],     # 连续变量
        'x1': [1, 3],     # 连续变量
        'x2': [2, 4]      # 连续变量
    }
    
    print("\nSEDSearchCV配置:")
    print("- 参数空间: x0 ∈ [0, 2], x1 ∈ [1, 3], x2 ∈ [2, 4]")
    print("- 粒子数量: 20")
    print("- 最大迭代次数: 50")
    print("- 交叉验证折数: 3")
    
    # 创建并运行搜索
    searcher = SEDSearchCV(
        estimator=dummy_model,
        param_space=param_space,
        n_particles=20,
        max_iter=50,
        cv=3
    )
    
    searcher.fit()
    
    # 输出结果
    print("\n最佳参数组合:", searcher.best_params_)
    print("最佳得分:", searcher.best_score_)
    
    # 绘制收敛曲线
    searcher.plot_convergence()

if __name__ == "__main__":
    print("开始SED优化器全功能测试")
    print("=" * 50)
    
    # 运行所有测试
    test_sedo_all_features()
    test_sedsearchcv()
    
    print("=" * 50)
    print("SED优化器全功能测试完成")