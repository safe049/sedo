import numpy as np
import time
from typing import List, Tuple, Callable, Optional, Union

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

def plot_convergence(sedo_history, pso_history):
    plt.figure(figsize=(10, 6))
    plt.plot(sedo_history, label='SEDO', linestyle='--', marker='.')
    plt.plot(pso_history, label='PSO', linestyle='-', marker='x')
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("Convergence Curve Comparison: SEDO vs PSO")
    plt.legend()
    plt.grid(True)
    plt.show()


# 主程序入口
if __name__ == "__main__":
    DIMENSIONS = 10
    MAX_ITER = 100
    BOUNDS = [(-5, 5)] * DIMENSIONS

    # 测试 Sphere 函数
    print("\n=== Testing on Sphere Function ===")
    
    # SEDO
    print("\nRunning SEDO...")
    sedo_opt = SEDOptimizer(
        objective_func=sphere,
        problem_dim=DIMENSIONS,
        n_particles=30,
        bounds=BOUNDS,
        use_parallel=True
    )
    sedo_opt.optimize(MAX_ITER)
    sedo_sol = sedo_opt.get_best_solution()
    sedo_fit = sedo_opt.global_best_fit
    sedo_hist = [h['best_fitness'] for h in sedo_opt.history if h['best_fitness'] is not None]

    # PSO
    print("\nRunning PSO...")
    pso_opt = PSOOptimizer(
        objective_func=sphere,
        problem_dim=DIMENSIONS,
        n_particles=30,
        bounds=BOUNDS,
        use_parallel=False
    )
    pso_opt.optimize(MAX_ITER)
    pso_sol = pso_opt.get_best_solution()
    pso_fit = pso_opt.gbest_fitness
    pso_hist = pso_opt.history

    print("\nSphere Function Results:")
    print(f"{'Algorithm':<10} {'Best Fitness':<15} {'Time Taken':<15}")
    print("-" * 40)
    print(f"{'SEDO':<10} {sedo_fit:<15.6f} N/A")
    print(f"{'PSO':<10} {pso_fit:<15.6f} N/A")

    plot_convergence(sedo_hist, pso_hist)


    # 测试 Rosenbrock 函数
    print("\n=== Testing on Rosenbrock Function ===")
    
    # SEDO
    print("\nRunning SEDO...")
    sedo_opt = SEDOptimizer(
        objective_func=rosenbrock,
        problem_dim=DIMENSIONS,
        n_particles=30,
        bounds=BOUNDS,
        use_parallel=True
        
    )
    sedo_opt.optimize(MAX_ITER)
    sedo_sol_r = sedo_opt.get_best_solution()
    sedo_fit_r = sedo_opt.global_best_fit
    sedo_hist_r = [h['best_fitness'] for h in sedo_opt.history if h['best_fitness'] is not None]

    # PSO
    print("\nRunning PSO...")
    pso_opt = PSOOptimizer(
        objective_func=rosenbrock,
        problem_dim=DIMENSIONS,
        n_particles=30,
        bounds=BOUNDS,
        use_parallel=False
    )
    pso_opt.optimize(MAX_ITER)
    pso_sol_r = pso_opt.get_best_solution()
    pso_fit_r = pso_opt.gbest_fitness
    pso_hist_r = pso_opt.history

    print("\nRosenbrock Function Results:")
    print(f"{'Algorithm':<10} {'Best Fitness':<15} {'Time Taken':<15}")
    print("-" * 40)
    print(f"{'SEDO':<10} {sedo_fit_r:<15.6f} N/A")
    print(f"{'PSO':<10} {pso_fit_r:<15.6f} N/A")

    plot_convergence(sedo_hist_r, pso_hist_r)


    # 测试 Ackley 函数
    print("\n=== Testing on Ackley Function ===")
    
    # SEDO
    print("\nRunning SEDO...")
    sedo_opt = SEDOptimizer(
        objective_func=ackley,
        problem_dim=DIMENSIONS,
        n_particles=30,
        bounds=BOUNDS,
        use_parallel=True
    )
    sedo_opt.optimize(MAX_ITER)
    sedo_sol_a = sedo_opt.get_best_solution()
    sedo_fit_a = sedo_opt.global_best_fit
    sedo_hist_a = [h['best_fitness'] for h in sedo_opt.history if h['best_fitness'] is not None]

    # PSO
    print("\nRunning PSO...")
    pso_opt = PSOOptimizer(
        objective_func=ackley,
        problem_dim=DIMENSIONS,
        n_particles=30,
        bounds=BOUNDS,
        use_parallel=False
    )
    pso_opt.optimize(MAX_ITER)
    pso_sol_a = pso_opt.get_best_solution()
    pso_fit_a = pso_opt.gbest_fitness
    pso_hist_a = pso_opt.history

    print("\nAckley Function Results:")
    print(f"{'Algorithm':<10} {'Best Fitness':<15} {'Time Taken':<15}")
    print("-" * 40)
    print(f"{'SEDO':<10} {sedo_fit_a:<15.6f} N/A")
    print(f"{'PSO':<10} {pso_fit_a:<15.6f} N/A")

    plot_convergence(sedo_hist_a, pso_hist_a)

    print("\nAll Tests Complete!")
    print("=" * 40)
    print("Total Test Results:")
    print(f"{'Function':<15} {'SEDO':<15} {'PSO':<15}")
    print("-" * 40)
    print(f"{'Sphere':<15} {sedo_fit:<15.6f} {pso_fit:<15.6f}")
    print(f"{'Rosenbrock':<15} {sedo_fit_r:<15.6f} {pso_fit_r:<15.6f}")
    print(f"{'Ackley':<15} {sedo_fit_a:<15.6f} {pso_fit_a:<15.6f}")
    print("=" * 40)
