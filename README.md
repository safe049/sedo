# Social Entropy Diffusion Optimization (SEDO)

Social Entropy Diffusion Optimization（SEDO）是一个受社会学、热力学和量子物理启发的新型智能优化算法。该算法模拟粒子在文化空间中的信息传播行为，通过熵流控制探索与开发的平衡，适用于求解复杂非线性、多模态或高维优化问题。

---

## 📦 特性

- **支持单目标与多目标优化**：可应用于Pareto前沿搜索
- **连续与离散变量混合优化**：支持整数/枚举型变量处理
- **并行计算支持**：使用`multiprocessing`提升适应度评估效率
- **自适应温度调节**：根据多样性动态调整系统温度
- **多样性监控与重启机制**：防止早熟收敛
- **初始化策略多样**：支持LHS、正交设计等初始化方法
- **结果保存与恢复**：支持checkpoint保存与加载
- **收敛曲线可视化**：绘制最优解变化过程

---

## 🧩 模块结构

```
sedo/
├── __init__.py
├── base.py                  # 基础接口定义
├── particle.py              # QuantumParticle类
├── optimizer.py             # SEDOptimizer核心逻辑
├── utils.py                 # 工具函数（保存/加载、绘图等）
└── search.py                # Scikit-learn风格封装器
```

---

## 🔧 安装

```bash
pip install numpy scipy matplotlib
```

将本项目代码保存为模块目录结构，并直接导入使用。

安装为开发包：

```bash
pip install -e .
```

---

## 🚀 快速开始

### 单目标优化示例：Sphere函数

```python
import numpy as np
from sedo.optimizer import SEDOptimizer

def sphere(x):
    return sum(xi ** 2 for xi in x)

if __name__ == "__main__":
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

    optimizer.optimize(max_iter=100)
    best_solution = optimizer.get_best_solution()
    print("\nOptimization Complete!")
    print(f"Best Solution Found: {best_solution}")
    print(f"Best Fitness Value: {optimizer.global_best_fit}")

    optimizer.plot_convergence()
```

### 多目标优化示例：ZDT1问题

```python
import numpy as np
from sedo.optimizer import SEDOptimizer

def zdt1(x):
    n = len(x)
    f1 = x[0]
    
    g = 1 + 9 * sum(x[1:]) / (n - 1)
    h = 1 - np.sqrt(f1 / g)
    
    f2 = g * h
    return [f1, f2]

if __name__ == "__main__":
    multi_optimizer = SEDOptimizer(
        objective_func=zdt1,
        problem_dim=10,
        n_particles=50,
        bounds=[(0, 1)] * 10,
        multi_objective=True,
        use_parallel=True
    )
    
    multi_optimizer.optimize(max_iter=200)
    pareto_front = multi_optimizer.get_best_solution()
    print(f"\nZDT1问题找到的Pareto前沿包含 {len(pareto_front)} 个解")
    
    # 绘制Pareto前沿
    import matplotlib.pyplot as plt
    
    fitnesses = [p.fitness for p in pareto_front]
    f1s = [f[0] for f in fitnesses]
    f2s = [f[1] for f in fitnesses]
    
    plt.scatter(f1s, f2s)
    plt.title("Approximated Pareto Front (ZDT1)")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.show()
```

### 使用Scikit-learn风格封装器

```python
from sedo.search import SEDSearchCV

def dummy_model(params):
    x0, x1, x2 = params
    return (x0 - 1) ** 2 + (x1 - 2) ** 2 + (x2 - 3) ** 2

param_space = {
    'x0': [0, 2],     # 连续变量
    'x1': [1, 3],     # 连续变量
    'x2': [2, 4]      # 连续变量
}

searcher = SEDSearchCV(
    estimator=dummy_model,
    param_space=param_space,
    n_particles=20,
    max_iter=50,
    cv=3
)

searcher.fit()

print("\n最佳参数组合:", searcher.best_params_)
print("最佳得分:", searcher.best_score_)

searcher.plot_convergence()
```

---

## 🧪 单元测试

我们提供了完整的单元测试以确保功能稳定性：

```bash
./test.sh
```

---

## 🛠️ 开发工具

我们建议使用以下工具进行开发：

- `pytest`: 用于运行单元测试
- `matplotlib`: 用于可视化结果
- `numpy`, `scipy`: 用于数值计算
- `multiprocessing`: 用于并行计算

---

## 📁 文件结构建议

```
sedo/
│
├── __init__.py
├── base.py                  # 基础接口定义
├── particle.py              # QuantumParticle类
├── optimizer.py             # SEDOptimizer核心逻辑
├── utils.py                 # 工具函数（保存/加载、绘图等）
└── search.py                # Scikit-learn风格封装器

setup.py
README.md
examples/
    example_sphere.py
tests/
    test_optimizer.py
```

---

## 📌 贡献指南

欢迎提交issue和PR！如果你有任何疑问、Bug报告或功能建议，请随时联系作者：
- Email: safe049@163.com
- GitHub: https://github.com/safe049/sedo

---

## 📝 许可证

该项目采用MIT许可证。详情请参阅LICENSE文件。