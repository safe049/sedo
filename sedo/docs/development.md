# SEDO 开发文档 | Social Entropy Diffusion Optimization (SEDO) Development Documentation

[[English]](#english-version) | [[中文]](#chinese-version)

<a id="chinese-version"></a>

## 1. 算法概述

社会熵扩散优化(Social Entropy Diffusion Optimization，SEDO)是一种受社会学、热力学和量子物理启发的新型智能优化算法。该算法模拟文化空间中信息传播行为，通过熵流机制控制探索与开发的平衡。

### 核心特点

- **量子化粒子表示**：每个粒子具有叠加态(探索/开发)
- **文化维度模型**：6维文化特征向量模拟社会传播
- **熵流机制**：基于热力学第二定律的搜索过程控制
- **动态坍缩**：根据熵阈值自动切换搜索模式
- **多目标支持**：内置Pareto前沿维护机制

## 2. 核心架构

### 2.1 主要组件

#### QuantumParticle类
```python
class QuantumParticle:
    def __init__(self, problem_dim: int, discrete_dims: Optional[List[int]] = None):
        self.cultural_dimension = np.random.normal(0, 1, 6)  # 6维文化特征
        self.entropy_phase = complex(random.random(), random.random())  # 熵相位(复数)
        self.positive_entropy = random.random()  # 正熵
        self.negative_entropy = random.random()  # 负熵
        self.superposition = [ExplorationState(), ExploitationState()]  # 量子叠加态
        self.position = None  # 当前位置
        self.velocity = None  # 当前速度
        self.fitness = float('inf')  # 适应度值
```

## 3. 核心算法流程

### 3.1 主优化循环
```python
def optimize(self, max_iter):
    for iter in range(max_iter):
        # 1. 评估适应度
        self._evaluate_fitness()
        
        # 2. 计算熵流
        for p in self.particles:
            flow = self._calculate_entropy_flow(p)
            p.positive_entropy += np.real(flow) * 0.01
            p.negative_entropy += np.imag(flow) * 0.01
```

## 4. 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| n_particles | int | 30 | 粒子数量 |
| barrier_height | float | 0.5 | 文化传播势垒高度 |

## 5. 使用方法

### 基础使用
```python
from sedo import SEDOptimizer

def sphere(x):
    return sum(x**2)

optimizer = SEDOptimizer(sphere, problem_dim=10)
optimizer.optimize(max_iter=100)
```

<a id="english-version"></a>
## Social Entropy Diffusion Optimization (SEDO) Development Documentation

[[English]](#english-version) | [[中文]](#chinese-version)

## 1. Algorithm Overview

Social Entropy Diffusion Optimization (SEDO) is a novel intelligent optimization algorithm inspired by sociology, thermodynamics and quantum physics. It simulates information diffusion in cultural space and controls exploration-exploitation balance through entropy flow mechanism.

### Key Features

- **Quantum particle representation**: Each particle has superposition states (exploration/exploitation)
- **Cultural dimension model**: 6D cultural feature vector for social diffusion
- **Entropy flow mechanism**: Search process control based on second law of thermodynamics
- **Dynamic collapse**: Automatic mode switching based on entropy threshold
- **Multi-objective support**: Built-in Pareto frontier maintenance

## 2. Core Architecture

### 2.1 Main Components

#### QuantumParticle Class
```python
class QuantumParticle:
    def __init__(self, problem_dim: int, discrete_dims: Optional[List[int]] = None):
        self.cultural_dimension = np.random.normal(0, 1, 6)  # 6D cultural features
        self.entropy_phase = complex(random.random(), random.random())  # Entropy phase (complex)
        self.positive_entropy = random.random()  # Positive entropy
        self.negative_entropy = random.random()  # Negative entropy
        self.superposition = [ExplorationState(), ExploitationState()]  # Quantum superposition
        self.position = None  # Current position
        self.velocity = None  # Current velocity
        self.fitness = float('inf')  # Fitness value
```

## 3. Core Algorithm Flow

### 3.1 Main Optimization Loop
```python
def optimize(self, max_iter):
    for iter in range(max_iter):
        # 1. Evaluate fitness
        self._evaluate_fitness()
        
        # 2. Calculate entropy flow
        for p in self.particles:
            flow = self._calculate_entropy_flow(p)
            p.positive_entropy += np.real(flow) * 0.01
            p.negative_entropy += np.imag(flow) * 0.01
```

## 4. Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_particles | int | 30 | Number of particles |
| barrier_height | float | 0.5 | Cultural diffusion barrier height |

## 5. Usage Examples

### Basic Usage
```python
from sedo import SEDOptimizer

def sphere(x):
    return sum(x**2)

optimizer = SEDOptimizer(sphere, problem_dim=10)
optimizer.optimize(max_iter=100)
```

[Back to Top](#sedo-开发文档--social-entropy-diffusion-optimization-sedo-development-documentation)
