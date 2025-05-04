# Social Entropy Diffusion Optimization (SEDO) Developer Documentation

---

## 📘 Overview

**Social Entropy Diffusion Optimization (SEDO)** is a novel intelligent optimization algorithm inspired by sociology, thermodynamics, and quantum physics. This algorithm simulates the information propagation behavior of particles in cultural space, balancing exploration and exploitation through entropy flow control. It is suitable for solving complex nonlinear, multimodal, or high-dimensional optimization problems.

---

## 🧩 1. Functional Features

| Feature | Description |
|--------|-------------|
| ✅ Single and Multi-objective Optimization | Supports Pareto front search |
| ✅ Continuous and Discrete Variable Hybrid Optimization | Handles integer/enumeration variables |
| ✅ Parallel Computing Support | Uses `multiprocessing` to speed up fitness evaluation |
| ✅ Adaptive Temperature Regulation | Dynamically adjusts system temperature based on diversity |
| ✅ Diversity Monitoring & Restart Mechanism | Prevents premature convergence |
| ✅ Diverse Initialization Strategies | Supports LHS, orthogonal design, etc. |
| ✅ Result Saving & Recovery | Supports checkpoint saving and loading |
| ✅ Convergence Curve Visualization | Plots the evolution of best solutions |

---

## 📦 2. Installation & Dependencies

### ⚙️ Environment Requirements:

- Python >= 3.8
- NumPy
- SciPy
- Matplotlib (optional, for plotting)
- Multiprocessing (built-in)

### 💾 Installation Instructions:

You can save the code as a module directory structure and import it directly for use:

```bash
pip install numpy scipy matplotlib
```

---

## 🧱 3. Module Structure

The project uses a modular organization for easier maintenance and extension.

```
sedo/
├── __init__.py
├── base.py                  # Base interface definitions
├── particle.py              # QuantumParticle class
├── optimizer.py             # Core logic of SEDOptimizer
├── utils.py                 # Utility functions (save/load, plotting, etc.)
└── search.py                # Scikit-learn style wrapper
```

---

## 📚 4. Core Classes & Methods

### 🧠 `QuantumParticle`

Represents a quantum-state particle with cultural dimensions and entropy values.

#### Attributes:

| Name | Type | Description |
|------|------|-------------|
| `cultural_dimension` | np.ndarray | Six-dimensional Hofstede cultural dimensions |
| `entropy_phase` | complex | Entropy phase in complex plane |
| `positive_entropy` | float | Exploration factor (positive entropy) |
| `negative_entropy` | float | Exploitation factor (negative entropy) |
| `position` | np.ndarray | Position in solution space |
| `velocity` | np.ndarray | Current velocity |
| `fitness` | float | Fitness value |
| `collapsed` | bool | Whether the state has collapsed |
| `state` | object | Current state (ExplorationState or ExploitationState) |

#### Methods:

| Method Name | Parameters | Description |
|------------|------------|-------------|
| `set_position(position, bounds)` | position: np.ndarray, bounds: List[Tuple[float, float]] | Sets position and handles discrete variables |
| `init_random_position(bounds, method='uniform')` | bounds: List[Tuple[float, float]], method: str | Randomly initializes position |

---

### 🤖 `SEDOptimizer`

Core optimizer class implementing the full SEDO algorithm.

#### Initialization Parameters:

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `objective_func` | Callable[[np.ndarray], Union[float, List[float]]] | - | Objective function |
| `problem_dim` | int | - | Problem dimensionality |
| `n_particles` | int | 30 | Number of particles |
| `barrier_height` | float | 0.5 | Quantum barrier height |
| `entropy_threshold` | float | 0.8 | Entropy collapse threshold |
| `temperature` | float | 1.0 | Initial system temperature |
| `bounds` | List[Tuple[float, float]] | [(-5,5)]*dim | Search range per dimension |
| `multi_objective` | bool | False | Enable multi-objective optimization |
| `use_parallel` | bool | True | Use parallel computation |
| `init_method` | str | 'uniform' | Initialization method ['uniform', 'lhs', 'orthogonal'] |
| `discrete_dims` | List[int] | None | Indices of discrete variables |

#### Main Methods:

| Method Name | Parameters | Description |
|------------|------------|-------------|
| `optimize(max_iter, callback=None)` | max_iter: int, callback: Optional[Callable] | Executes optimization process |
| `get_best_solution()` | - | Returns current best solution |
| `save_checkpoint(file_path)` | file_path: str | Saves optimizer state |
| `load_checkpoint(file_path)` | file_path: str | Loads optimizer state |
| `export_results(file_path, fmt='json')` | file_path: str, fmt: str | Exports results to file |
| `plot_convergence()` | - | Plots convergence curve |
| `plot_distribution()` | - | Plots final particle distribution (only for 2D/3D) |

---

### 🔍 `SEDSearchCV` (Scikit-learn Style Wrapper)

Provides an interface similar to `GridSearchCV`, convenient for hyperparameter tuning.

#### Initialization Parameters:

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `estimator` | Callable[[np.ndarray], float] | - | Estimation function |
| `param_space` | Dict[str, List[float]] | - | Parameter space |
| `n_particles` | int | 30 | Number of particles |
| `max_iter` | int | 100 | Max iterations |
| `scoring` | Callable | lambda x: -x | Scoring function |
| `cv` | int | 3 | Cross-validation folds |
| `verbose` | int | 1 | Verbosity level |

#### Main Attributes:

| Attribute | Type | Description |
|----------|------|-------------|
| `best_params_` | Dict[str, Any] | Best parameter combination |
| `best_score_` | float | Best score achieved |
| `optimizer_` | SEDOptimizer | Internal optimizer instance |

#### Example:

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

## 🛠️ 5. Example Program: Sphere Function Optimization

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

## 🖥 6. Unit Tests

The project includes comprehensive unit test cases. You can run them using:

```bash
./test.sh
```

---

## 📈 7. Performance Optimization Suggestions

| Technique | Description |
|----------|-------------|
| Async Update | Use coroutines for asynchronous state updates |
| Caching | Cache recently visited fitness values to avoid recomputation |
| Distributed Computation | Use Dask / Ray for distributed optimization |

---

## 📌 8. Future Expansion Directions

| Direction | Description |
|----------|-------------|
| ✅ Reinforcement Learning Integration | Model particle behavior as RL policy |
| ✅ Web API Interface | Provide RESTful API using Flask/FastAPI |
| ✅ GUI Interface | Build graphical interface using PyQt5/Tkinter |
| ✅ Auto-Tuning Module | Integrate Bayesian optimization for adaptive hyperparameters |
| ✅ Time Series Prediction Optimization | Specialized optimization for LSTM, Transformer model parameters |

---

## 📞 Contact & Feedback

If you have any questions, bug reports, or feature suggestions, please feel free to contact the author:
- Email: safe049@163.com
- GitHub: https://github.com/safe049/sedo