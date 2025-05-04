import numpy as np
import time
import statistics
from typing import List, Tuple, Callable, Optional, Union
import matplotlib.pyplot as plt

from sedo.optimizer import SEDOptimizer 

# 自定义 PSO 实现（用于对比）
class PSOOptimizer:
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        problem_dim: int,
        n_particles: int = 30,
        bounds: Optional[List[Tuple[float, float]]] = None,
        use_parallel: bool = False
    ):
        self.objective_func = objective_func
        self.problem_dim = problem_dim
        self.n_particles = n_particles
        self.bounds = bounds or [(-5, 5)] * problem_dim
        self.use_parallel = use_parallel

        # 初始化粒子位置和速度
        self.positions = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(n_particles, problem_dim)
        )
        self.velocities = np.random.uniform(-1, 1, (n_particles, problem_dim))
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.array([float('inf')] * n_particles)
        self.gbest_position = None
        self.gbest_fitness = float('inf')
        self.history = []

    def _evaluate(self, x):
        return self.objective_func(x)

    def optimize(self, max_iter: int):
        start_time = time.time()

        for iter in range(max_iter):
            # 评估适应度
            if self.use_parallel:
                from multiprocessing import Pool, cpu_count
                with Pool(cpu_count()) as pool:
                    fitnesses = pool.map(self._evaluate, self.positions)
            else:
                fitnesses = [self._evaluate(pos) for pos in self.positions]

            # 更新个体最优
            for i in range(self.n_particles):
                if fitnesses[i] < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitnesses[i]
                    self.pbest_positions[i] = self.positions[i].copy()

                if self.pbest_fitness[i] < self.gbest_fitness:
                    self.gbest_fitness = self.pbest_fitness[i]
                    self.gbest_position = self.pbest_positions[i].copy()

            # 更新速度与位置
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_weight = 1.5
            social_weight = 1.5
            inertia_weight = 0.7

            for i in range(self.n_particles):
                self.velocities[i] = (
                    inertia_weight * self.velocities[i] +
                    cognitive_weight * r1 * (self.pbest_positions[i] - self.positions[i]) +
                    social_weight * r2 * (self.gbest_position - self.positions[i])
                )

                self.positions[i] = np.clip(
                    self.positions[i] + self.velocities[i],
                    [b[0] for b in self.bounds],
                    [b[1] for b in self.bounds]
                )

            self.history.append(self.gbest_fitness)
            print(f"Iteration {iter+1}/{max_iter}, Best Fitness: {self.gbest_fitness:.4f}")

        print(f"PSO Optimization Complete! Time: {time.time() - start_time:.2f}s")

    def get_best_solution(self) -> np.ndarray:
        return self.gbest_position

    def get_convergence_history(self):
        return self.history


# 测试函数
def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) + (1 - x[:-1]) ** 2))

def ackley(x: np.ndarray) -> float:
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    return float(-a * np.exp(-b * np.sqrt((1/n) * np.sum(x ** 2))) - 
                np.exp((1/n) * np.sum(np.cos(c * x))) + a + np.e)

def rastrigin(x: np.ndarray) -> float:
    return float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10))

def griewank(x: np.ndarray) -> float:
    product = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return float(1 + np.sum(x ** 2) / 4000 - product)

def schwefel(x: np.ndarray) -> float:
    return float(-np.sum(x * np.sin(np.sqrt(np.abs(x)))))

def levy(x: np.ndarray) -> float:
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return float(term1 + term2 + term3)

def zakharov(x: np.ndarray) -> float:
    idx = np.arange(1, len(x)+1)
    return float(np.sum(x ** 2) + np.sum(0.5 * idx * x) ** 2 + np.sum(0.5 * idx * x) ** 4)

def michalewicz(x: np.ndarray, m: int = 10) -> float:
    return float(-np.sum(np.sin(x) * np.sin((np.arange(1, len(x)+1) * x ** 2) / np.pi) ** (2 * m)))

def schwefel_12(x: np.ndarray) -> float:
    sum_ = 0
    for i in range(len(x)):
        sum_ += abs(x[:i+1]).sum()
    return sum_

# 设置随机种子以保证实验可复现性
def set_seed(seed=42):
    np.random.seed(seed)

# 对齐历史长度
def align_histories(histories):
    max_len = max(len(h) for h in histories)
    aligned = []
    for h in histories:
        if len(h) < max_len:
            last_val = h[-1]
            h += [last_val] * (max_len - len(h))
        aligned.append(h)
    return aligned

# 绘制收敛曲线
def plot_convergence(sedo_histories, pso_histories, title):
    sedo_histories = align_histories(sedo_histories)
    pso_histories = align_histories(pso_histories)
    
    avg_sedo = np.mean(sedo_histories, axis=0)
    avg_pso = np.mean(pso_histories, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(avg_sedo, label='SEDO', linestyle='--', marker='.')
    plt.plot(avg_pso, label='PSO', linestyle='-', marker='x')
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title(f"Convergence Curve Comparison: {title}")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    DIMENSIONS = 10
    MAX_ITER = 100
    BOUNDS = [(-5, 5)] * DIMENSIONS
    N_RUNS = 5

    def michalewicz_wrapper(x):
        return michalewicz(x)

    test_functions = [
        ('Sphere', sphere),
        ('Rosenbrock', rosenbrock),
        ('Ackley', ackley),
        ('Rastrigin', rastrigin),
        ('Griewank', griewank),
        ('Schwefel', schwefel),
        ('Levy', levy),
        ('Zakharov', zakharov),
        ('Michalewicz', michalewicz_wrapper),
        ('Schwefel_1.2', schwefel_12)
    ]

    results_table = []

    # 新增：用于保存所有函数的平均历史
    all_sedo_histories = []
    all_pso_histories = []

    for func_name, func in test_functions:
        print(f"\n=== Testing on {func_name} Function ===")
        sedo_fitnesses = []
        pso_fitnesses = []
        sedo_histories = []
        pso_histories = []

        for run in range(N_RUNS):
            seed = run * 100
            set_seed(seed)
            print(f"\nRun {run+1}/{N_RUNS} (Seed={seed})")

            # SEDO
            print("\nRunning SEDO...")
            sedo_opt = SEDOptimizer(
                objective_func=func,
                problem_dim=DIMENSIONS,
                n_particles=30,
                bounds=BOUNDS,
                use_parallel=True
            )
            sedo_opt.optimize(MAX_ITER)
            sedo_fitnesses.append(sedo_opt.global_best_fit)
            sedo_histories.append([h['best_fitness'] for h in sedo_opt.history if h['best_fitness'] is not None])

            # PSO
            print("\nRunning PSO...")
            pso_opt = PSOOptimizer(
                objective_func=func,
                problem_dim=DIMENSIONS,
                n_particles=30,
                bounds=BOUNDS,
                use_parallel=False
            )
            pso_opt.optimize(MAX_ITER)
            pso_fitnesses.append(pso_opt.gbest_fitness)
            pso_histories.append(pso_opt.get_convergence_history())

        # 计算平均和标准差
        avg_sedo = statistics.mean(sedo_fitnesses)
        std_sedo = statistics.stdev(sedo_fitnesses) if len(sedo_fitnesses) > 1 else 0
        avg_pso = statistics.mean(pso_fitnesses)
        std_pso = statistics.stdev(pso_fitnesses) if len(pso_fitnesses) > 1 else 0

        # 成功率计算（根据阈值）
        success_thresholds = {
            'Sphere': 1e-4,
            'Rosenbrock': 1e-2,
            'Ackley': 1e-2,
            'Rastrigin': 1e-1,
            'Griewank': 1e-3,
            'Zakharov': 1e-3,
            'Michalewicz': 1e-1
        }
        sr_sedo = sum(1 for f in sedo_fitnesses if f <= success_thresholds.get(func_name, float('inf'))) / N_RUNS
        sr_pso = sum(1 for f in pso_fitnesses if f <= success_thresholds.get(func_name, float('inf'))) / N_RUNS

        results_table.append({
            "Function": func_name,
            "Algorithm": "SEDO",
            "Avg Fitness": avg_sedo,
            "Std Dev": std_sedo,
            "Success Rate": sr_sedo
        })
        results_table.append({
            "Function": func_name,
            "Algorithm": "PSO",
            "Avg Fitness": avg_pso,
            "Std Dev": std_pso,
            "Success Rate": sr_pso
        })

        # 保存平均历史用于最终绘图
        avg_sedo_history = np.mean(align_histories(sedo_histories), axis=0).tolist()
        avg_pso_history = np.mean(align_histories(pso_histories), axis=0).tolist()
        all_sedo_histories.append(avg_sedo_history)
        all_pso_histories.append(avg_pso_history)

        # 绘图：每个函数单独一张图
        #plot_convergence(sedo_histories, pso_histories, f"{func_name} Function (Average of {N_RUNS} runs)")


    # 输出表格
    print("\nAll Tests Complete!")
    print("=" * 80)
    print(f"{'Function':<15} {'Algorithm':<10} {'Avg Fitness':<15} {'Std Dev':<15} {'SR (%)':<10}")
    print("-" * 80)
    for row in results_table:
        print(f"{row['Function']:<15} {row['Algorithm']:<10} "
              f"{row['Avg Fitness']:<15.6f} {row['Std Dev']:<15.6f} {row['Success Rate']*100:<10.2f}")
    print("=" * 80)