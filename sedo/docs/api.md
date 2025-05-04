以下是为该算法编写的完善API文档，包含所有开发用函数、方法、类及其参数说明：

---

# SEDO (Social Entropy Diffusion Optimization) API 文档

## 1. 模块结构
- **base.py**: 定义优化器的抽象基类。
- **optimizer.py**: 实现SEDO优化器的核心逻辑。
- **particle.py**: 定义量子粒子及其状态。
- **search.py**: 提供基于SEDO的超参数搜索工具。
- **utils.py**: 提供辅助工具，如绘图、保存和加载检查点等。

## 2. 类与函数

### 2.1 `BaseOptimizer` (抽象基类)
**文件**: `base.py`

**描述**: 定义优化器的基本接口。

**方法**:
- `optimize(self, max_iter: int) -> None`: 执行优化过程。
  - **参数**:
    - `max_iter`: 最大迭代次数。
- `get_best_solution(self) -> Any`: 获取当前最优解。

### 2.2 `SEDOptimizer`
**文件**: `optimizer.py`

**描述**: SEDO优化器的核心实现，模拟文化空间中的信息传播行为，通过熵流机制控制探索与开发的平衡。

**初始化参数**:
- `objective_func`: 目标函数，输入为`np.ndarray`，输出为浮点数或列表（多目标）。
- `problem_dim`: 问题维度。
- `n_particles`: 粒子数量，默认为30。
- `barrier_height`: 势垒高度，影响文化传播距离衰减，默认为0.5。
- `entropy_threshold`: 熵阈值，用于控制探索与开发的切换，默认为0.7。
- `temperature`: 初始系统温度，默认为1.0。
- `bounds`: 各维度的搜索范围，默认为`[(-5, 5)] * problem_dim`。
- `multi_objective`: 是否启用多目标优化，默认为`False`。
- `use_parallel`: 是否使用并行计算，默认为`True`。
- `init_method`: 初始化方法，可选`['uniform', 'lhs', 'orthogonal']`，默认为`'uniform'`。
- `discrete_dims`: 离散变量的索引列表，默认为`None`。

**方法**:
- `optimize(self, max_iter: int, callback: Optional[Callable] = None) -> None`: 执行优化过程。
  - **参数**:
    - `max_iter`: 最大迭代次数。
    - `callback`: 回调函数，用于监控优化过程。
- `get_best_solution(self) -> Any`: 获取当前最优解。
  - **返回值**:
    - 单目标优化时返回最优位置（`np.ndarray`），多目标优化时返回非支配解集（`np.ndarray`列表）。

**内部方法**:
- `_evaluate_fitness(self)`: 评估粒子的适应度。
- `_is_dominated(self, fit1, fit2)`: 判断`fit1`是否支配`fit2`（多目标优化用）。
- `_update_pareto_front(self, particle)`: 更新Pareto前沿并维护外部档案。
- `_cultural_similarity(self, p1, p2)`: 计算两个粒子的文化相似度（余弦相似度）。
- `_calculate_entropy_flow(self, particle, neighbors)`: 计算熵流。
- `_dynamic_collapse(self)`: 动态坍缩机制，根据熵值决定粒子的探索/开发状态。
- `_cultural_crossover(self, p1, p2)`: 文化维度交叉操作。
- `_update_bass_model(self, curr_iter, max_iter)`: 更新Bass扩散模型的创新与模仿系数。
- `_calculate_diversity(self)`: 计算种群多样性。
- `_update_diversity(self)`: 记录多样性变化。
- `_restart_if_stagnant(self, threshold: float = 0.001, window: int = 5)`: 多样性重启机制。
- `_fine_tune_development(self, particle)`: 局部精细开发策略。

### 2.3 `QuantumParticle`
**文件**: `particle.py`

**描述**: 表示一个量子态粒子，具有文化维度、熵值、位置等属性。

**初始化参数**:
- `problem_dim`: 问题维度。
- `discrete_dims`: 离散变量的索引列表，默认为`None`。

**属性**:
- `cultural_dimension`: 文化维度（`np.ndarray`）。
- `entropy_phase`: 熵相位（复数）。
- `positive_entropy`: 正熵值。
- `negative_entropy`: 负熵值。
- `position`: 粒子位置（`np.ndarray`）。
- `velocity`: 粒子速度（`np.ndarray`）。
- `fitness`: 粒子适应度。
- `collapsed`: 是否坍缩。
- `state`: 粒子状态（`ExplorationState`或`ExploitationState`）。

**方法**:
- `set_position(self, position: np.ndarray, bounds: List[Tuple[float, float]]) -> None`: 设置粒子位置并进行边界约束。
- `init_random_position(self, bounds: List[Tuple[float, float]], method: str = 'uniform') -> None`: 初始化粒子位置。
  - **参数**:
    - `bounds`: 搜索范围。
    - `method`: 初始化方法，可选`['uniform', 'lhs', 'orthogonal']`。

### 2.4 `SEDSearchCV`
**文件**: `search.py`

**描述**: 基于SEDO的超参数搜索工具，用于机器学习模型的参数优化。

**初始化参数**:
- `estimator`: 机器学习模型的评估函数，输入为参数向量，输出为评估分数。
- `param_space`: 参数空间，字典形式，键为参数名，值为参数取值范围。
- `n_particles`: 粒子数量，默认为30。
- `max_iter`: 最大迭代次数，默认为100。
- `scoring`: 评分函数，默认为负值（用于最小化目标函数）。
- `cv`: 交叉验证次数，默认为3。
- `verbose`: 日志输出级别，默认为1。

**方法**:
- `fit(self, X=None, y=None) -> None`: 执行超参数搜索。
  - **参数**:
    - `X`: 训练数据（可选）。
    - `y`: 目标值（可选）。
- `plot_convergence(self) -> None`: 绘制收敛曲线。

### 2.5 辅助函数
**文件**: `utils.py`

**函数**:
- `plot_convergence(history: List[Dict[str, float]]) -> None`: 绘制收敛曲线。
  - **参数**:
    - `history`: 优化过程的历史记录，包含每一代的最优适应度等信息。
- `save_checkpoint(optimizer, file_path: str) -> None`: 保存优化器的状态到文件。
  - **参数**:
    - `optimizer`: 优化器实例。
    - `file_path`: 保存路径。
- `load_checkpoint(file_path: str) -> dict`: 从文件加载优化器的状态。
  - **参数**:
    - `file_path`: 文件路径。
  - **返回值**: 优化器的状态字典。
- `export_results(optimizer, file_path: str, fmt: str = 'json') -> None`: 导出优化结果。
  - **参数**:
    - `optimizer`: 优化器实例。
    - `file_path`: 输出文件路径。
    - `fmt`: 输出格式，可选`['json', 'csv']`。

---

## 3. 示例代码

### 3.1 使用`SEDOptimizer`优化单目标函数
```python
import numpy as np
from optimizer import SEDOptimizer

# 定义目标函数
def objective_function(x):
    return np.sum(x**2)

# 初始化优化器
optimizer = SEDOptimizer(
    objective_func=objective_function,
    problem_dim=5,
    n_particles=30,
    bounds=[(-10, 10)] * 5
)

# 执行优化
optimizer.optimize(max_iter=100)

# 获取最优解
best_solution = optimizer.get_best_solution()
print("Best Solution:", best_solution)
```

### 3.2 使用`SEDSearchCV`进行超参数搜索
```python
from search import SEDSearchCV

# 定义评估函数
def estimator(params):
    # 示例：简单的目标函数
    return np.sum(params**2)

# 定义参数空间
param_space = {
    'param1': [0, 1, 2, 3],
    'param2': [0.1, 0.2, 0.3, 0.4],
    'param3': [10, 20, 30, 40]
}

# 初始化搜索工具
search_cv = SEDSearchCV(
    estimator=estimator,
    param_space=param_space,
    n_particles=20,
    max_iter=50
)

# 执行搜索
search_cv.fit()

# 输出最优参数
print("Best Parameters:", search_cv.best_params_)
print("Best Score:", search_cv.best_score_)
```

---

## 4. 注意事项
- 在多目标优化中，`objective_func`应返回一个列表，表示多个目标的值。
- 使用并行计算时，确保目标函数可以被序列化。
- 重启机制和动态坍缩机制依赖于多样性监控，因此在某些情况下可能需要调整相关参数以避免过早收敛或陷入局部最优。

---

## 5. 版本信息
- **版本**: 1.0.0
- **最后更新**: 2025年5月4日

---
