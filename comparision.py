import numpy as np
import time
from typing import List, Tuple, Callable, Optional, Union
import statistics

# 引入你的 SEDO 算法
from sedo.optimizer import SEDOptimizer

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

# 自定义 PSO 实现
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


# 绘制收敛曲线
import matplotlib.pyplot as plt

def plot_convergence(sedo_history, pso_history, title):
    plt.figure(figsize=(10, 6))
    plt.plot(sedo_history, label='SEDO', linestyle='--', marker='.')
    plt.plot(pso_history, label='PSO', linestyle='-', marker='x')
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title(f"Convergence Curve Comparison: {title}")
    plt.legend()
    plt.grid(True)
    plt.show()


# 主程序入口
if __name__ == "__main__":
    DIMENSIONS = 10
    MAX_ITER = 100
    BOUNDS = [(-5, 5)] * DIMENSIONS
    N_RUNS = 5  # 每个测试运行5次

    def run_multiple_tests(objective_func, func_name):
        sedo_fitnesses = []
        pso_fitnesses = []
        sedo_histories = []
        pso_histories = []
        
        print(f"\n=== Testing on {func_name} Function ===")
        
        for run in range(N_RUNS):
            print(f"\nRun {run+1}/{N_RUNS}")
            
            # SEDO
            print("\nRunning SEDO...")
            sedo_opt = SEDOptimizer(
                objective_func=objective_func,
                problem_dim=DIMENSIONS,
                n_particles=30,
                bounds=BOUNDS,
                use_parallel=True
            )
            sedo_opt.optimize(MAX_ITER)
            sedo_fitnesses.append(sedo_opt.global_best_fit)
            sedo_hist = [h['best_fitness'] for h in sedo_opt.history if h['best_fitness'] is not None]
            sedo_histories.append(sedo_hist)
            
            # PSO
            print("\nRunning PSO...")
            pso_opt = PSOOptimizer(
                objective_func=objective_func,
                problem_dim=DIMENSIONS,
                n_particles=30,
                bounds=BOUNDS,
                use_parallel=False
            )
            pso_opt.optimize(MAX_ITER)
            pso_fitnesses.append(pso_opt.gbest_fitness)
            pso_histories.append(pso_opt.history)
        
        # 计算平均适应度和收敛曲线
        avg_sedo_fit = statistics.mean(sedo_fitnesses)
        avg_pso_fit = statistics.mean(pso_fitnesses)
        
        # 计算平均收敛曲线
        min_length = min(len(h) for h in sedo_histories)
        avg_sedo_hist = [statistics.mean([h[i] for h in sedo_histories]) for i in range(min_length)]
        
        min_length = min(len(h) for h in pso_histories)
        avg_pso_hist = [statistics.mean([h[i] for h in pso_histories]) for i in range(min_length)]
        
        print(f"\n{func_name} Function Results (Average of {N_RUNS} runs):")
        print(f"{'Algorithm':<10} {'Avg Best Fitness':<15} {'Std Dev':<15}")
        print("-" * 40)
        print(f"{'SEDO':<10} {avg_sedo_fit:<15.6f} {statistics.stdev(sedo_fitnesses):<15.6f}")
        print(f"{'PSO':<10} {avg_pso_fit:<15.6f} {statistics.stdev(pso_fitnesses):<15.6f}")
        
        plot_convergence(avg_sedo_hist, avg_pso_hist, f"{func_name} Function (Average of {N_RUNS} runs)")
        
        return avg_sedo_fit, avg_pso_fit, statistics.stdev(sedo_fitnesses), statistics.stdev(pso_fitnesses)
    
    # 运行所有测试
    sphere_sedo, sphere_pso, sphere_sedo_std, sphere_pso_std = run_multiple_tests(sphere, "Sphere")
    rosenbrock_sedo, rosenbrock_pso, rosenbrock_sedo_std, rosenbrock_pso_std = run_multiple_tests(rosenbrock, "Rosenbrock")
    ackley_sedo, ackley_pso, ackley_sedo_std, ackley_pso_std = run_multiple_tests(ackley, "Ackley")

    print("\nAll Tests Complete!")
    print("=" * 60)
    print("Total Test Results (Average of 5 runs):")
    print(f"{'Function':<15} {'Algorithm':<10} {'Avg Fitness':<15} {'Std Dev':<15}")
    print("-" * 60)
    print(f"{'Sphere':<15} {'SEDO':<10} {sphere_sedo:<15.6f} {sphere_sedo_std:<15.6f}")
    print(f"{'Sphere':<15} {'PSO':<10} {sphere_pso:<15.6f} {sphere_pso_std:<15.6f}")
    print(f"{'Rosenbrock':<15} {'SEDO':<10} {rosenbrock_sedo:<15.6f} {rosenbrock_sedo_std:<15.6f}")
    print(f"{'Rosenbrock':<15} {'PSO':<10} {rosenbrock_pso:<15.6f} {rosenbrock_pso_std:<15.6f}")
    print(f"{'Ackley':<15} {'SEDO':<10} {ackley_sedo:<15.6f} {ackley_sedo_std:<15.6f}")
    print(f"{'Ackley':<15} {'PSO':<10} {ackley_pso:<15.6f} {ackley_pso_std:<15.6f}")
    print("=" * 60)