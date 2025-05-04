# Social Entropy Diffusion Optimization (SEDO) 开发文档

---

## 📘 概述

**Social Entropy Diffusion Optimization (SEDO)** 是一个受社会学、热力学与量子物理启发的新型智能优化算法。该算法模拟粒子在文化空间中的信息传播行为，通过熵流控制探索与开发的平衡，适用于求解复杂非线性、多模态或高维优化问题。

---

## 🧩 1. 功能特性

| 特性 | 描述 |
|------|------|
| ✅ 支持单目标与多目标优化 | 可用于 Pareto 前沿搜索 |
| ✅ 连续与离散变量混合优化 | 支持整数/枚举型变量处理 |
| ✅ 并行计算支持 | 使用 `multiprocessing` 提升适应度评估效率 |
| ✅ 自适应温度调节 | 根据多样性动态调整系统温度 |
| ✅ 多样性监控与重启机制 | 防止早熟收敛 |
| ✅ 初始化策略多样 | 支持 LHS、正交设计等初始化方法 |
| ✅ 结果保存与恢复 | 支持 checkpoint 保存与加载 |
| ✅ 收敛曲线可视化 | 绘制最优解变化过程 |

---

## 📦 2. 安装与依赖

### ⚙️ 环境要求：

- Python >= 3.8
- NumPy
- SciPy
- Matplotlib（可选，用于绘图）
- Multiprocessing（内置）

### 💾 安装方式：

你可以将代码保存为模块目录结构，并直接导入使用：

```bash
pip install numpy scipy matplotlib
```

---

## 🧱 3. 模块结构说明

项目采用模块化组织方式，便于维护和扩展。

```
sedo/
├── __init__.py
├── base.py                  # 基础接口定义
├── particle.py              # QuantumParticle 类
├── optimizer.py             # SEDOptimizer 核心逻辑
├── utils.py                 # 工具函数（保存/加载、绘图等）
└── search.py                # Scikit-learn 风格封装器
```

---

## 📚 4. 核心类与方法说明

### 🧠 `QuantumParticle`

表示一个具有文化维度和熵值的量子态粒子。

#### 属性：

| 名称 | 类型 | 描述 |
|------|------|------|
| `cultural_dimension` | np.ndarray | 六维霍夫斯泰德文化维度 |
| `entropy_phase` | complex | 熵相位复平面表示 |
| `positive_entropy` | float | 探索因子（正熵） |
| `negative_entropy` | float | 开发因子（负熵） |
| `position` | np.ndarray | 解空间位置 |
| `velocity` | np.ndarray | 当前速度 |
| `fitness` | float | 适应度值 |
| `collapsed` | bool | 是否已坍缩 |
| `state` | object | 当前状态（ExplorationState 或 ExploitationState） |

#### 方法：

| 方法名 | 参数 | 描述 |
|--------|------|------|
| `set_position(position, bounds)` | position: np.ndarray, bounds: List[Tuple[float, float]] | 设置位置并处理离散变量 |
| `init_random_position(bounds, method='uniform')` | bounds: List[Tuple[float, float]], method: str | 随机初始化位置 |

---

### 🤖 `SEDOptimizer`

核心优化器类，实现完整的 SEDO 算法流程。

#### 初始化参数：

| 参数名 | 类型 | 默认值 | 描述 |
|-------|------|--------|------|
| `objective_func` | Callable[[np.ndarray], Union[float, List[float]]] | - | 目标函数 |
| `problem_dim` | int | - | 问题维度 |
| `n_particles` | int | 30 | 粒子数量 |
| `barrier_height` | float | 0.5 | 量子势垒高度 |
| `entropy_threshold` | float | 0.8 | 熵坍缩阈值 |
| `temperature` | float | 1.0 | 初始系统温度 |
| `bounds` | List[Tuple[float, float]] | [(-5,5)]*dim | 各维度搜索范围 |
| `multi_objective` | bool | False | 是否启用多目标优化 |
| `use_parallel` | bool | True | 是否使用并行计算 |
| `init_method` | str | 'uniform' | 初始化方法 ['uniform', 'lhs', 'orthogonal'] |
| `discrete_dims` | List[int] | None | 离散变量索引列表 |

#### 主要方法：

| 方法名 | 参数 | 描述 |
|--------|------|------|
| `optimize(max_iter, callback=None)` | max_iter: int, callback: Optional[Callable] | 执行优化流程 |
| `get_best_solution()` | - | 返回当前最优解 |
| `save_checkpoint(file_path)` | file_path: str | 保存当前优化器状态 |
| `load_checkpoint(file_path)` | file_path: str | 加载优化器状态 |
| `export_results(file_path, fmt='json')` | file_path: str, fmt: str | 导出结果到文件 |
| `plot_convergence()` | - | 绘制收敛曲线 |
| `plot_distribution()` | - | 绘制粒子最终分布（仅限 2D 和 3D） |

---

### 🔍 `SEDSearchCV`（Scikit-learn 风格封装）

提供类似 `GridSearchCV` 的接口，方便参数调优。

#### 初始化参数：

| 参数名 | 类型 | 默认值 | 描述 |
|-------|------|--------|------|
| `estimator` | Callable[[np.ndarray], float] | - | 估计函数 |
| `param_space` | Dict[str, List[float]] | - | 参数空间 |
| `n_particles` | int | 30 | 粒子数量 |
| `max_iter` | int | 100 | 最大迭代次数 |
| `scoring` | Callable | lambda x: -x | 评分函数 |
| `cv` | int | 3 | 交叉验证折数 |
| `verbose` | int | 1 | 输出详细信息等级 |

#### 主要属性：

| 属性名 | 类型 | 描述 |
|--------|------|------|
| `best_params_` | Dict[str, Any] | 最佳参数组合 |
| `best_score_` | float | 最佳得分 |
| `optimizer_` | SEDOptimizer | 内部使用的优化器实例 |

#### 示例：

```python
from sedo.search import SEDSearchCV

def sphere(x):
    return sum(xi ** 2 for xi in x)

param_space = {
    'x0': [-5, 5],
    'x1': [-5, 5],
    'x2': [-5, 5]
}

searcher = SEDSearchCV(sphere, param_space, n_particles=30, max_iter=100)
searcher.fit()

print("Best Params:", searcher.best_params_)
print("Best Score:", searcher.best_score_)
searcher.plot_convergence()
```

---

## 🛠️ 5. 示例程序：Sphere 函数优化

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

---

## 🖥 6. 单元测试
项目提供了完整的单元测试用例，你可以通过以下命令运行：

```bash
./test.sh
```

---

## 📈 7. 性能优化建议

| 技术 | 描述 |
|------|------|
| 异步更新 | 使用协程异步更新粒子状态 |
| 缓存机制 | 缓存最近访问的适应度值避免重复计算 |
| 分布式计算 | 使用 Dask / Ray 实现分布式优化 |

---

## 📌 8. 未来扩展方向

| 方向 | 描述 |
|------|------|
| ✅ 强化学习结合 | 将粒子行为建模为强化学习策略 |
| ✅ Web API 接口 | 使用 Flask/FastAPI 提供 RESTful 接口 |
| ✅ 图形界面 | 使用 PyQt5/Tkinter 构建图形化界面 |
| ✅ 自动调参模块 | 集成贝叶斯优化进行超参数自适应 |
| ✅ 时间序列预测优化 | 专门优化 LSTM、Transformer 等模型参数 |

---

## 📞 联系与反馈

如果你有任何疑问、Bug 报告或功能建议，请随时联系作者：
- Email: safe049@163.com
- GitHub: https://github.com/safe049/sedo

---