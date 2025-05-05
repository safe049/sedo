# gwo_optimizer.py
import numpy as np
from typing import List, Tuple, Callable, Optional, Union
from concurrent.futures import ProcessPoolExecutor
import time

class GWOOptimizer:
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

        # 初始化种群
        self.positions = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(n_particles, problem_dim)
        )
        self.fitnesses = np.array([float('inf')] * n_particles)

        self.alpha_pos = None
        self.alpha_fit = float('inf')
        self.beta_pos = None
        self.beta_fit = float('inf')
        self.delta_pos = None
        self.delta_fit = float('inf')

        self.best_position = None
        self.best_fitness = float('inf')
        self.history = []

    def _evaluate(self, x):
        return self.objective_func(x)

    def optimize(self, max_iter: int):
        start_time = time.time()

        for iter in range(max_iter):
            a = 2 - iter * (2 / max_iter)  # 线性递减参数

            if self.use_parallel:
                with ProcessPoolExecutor() as executor:
                    fits = list(executor.map(self._evaluate, self.positions))
            else:
                fits = [self._evaluate(pos) for pos in self.positions]

            # 更新 α, β, δ
            for i in range(self.n_particles):
                fit = fits[i]
                if fit < self.alpha_fit:
                    self.delta_fit = self.beta_fit
                    self.delta_pos = self.beta_pos
                    self.beta_fit = self.alpha_fit
                    self.beta_pos = self.alpha_pos
                    self.alpha_fit = fit
                    self.alpha_pos = self.positions[i].copy()
                elif fit < self.beta_fit:
                    self.delta_fit = self.beta_fit
                    self.delta_pos = self.beta_pos
                    self.beta_fit = fit
                    self.beta_pos = self.positions[i].copy()
                elif fit < self.delta_fit:
                    self.delta_fit = fit
                    self.delta_pos = self.positions[i].copy()

            self.best_fitness = self.alpha_fit
            self.best_position = self.alpha_pos

            # 更新位置
            for i in range(self.n_particles):
                r1 = np.random.rand(self.problem_dim)
                r2 = np.random.rand(self.problem_dim)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha_pos - self.positions[i])
                X1 = self.alpha_pos - A1 * D_alpha

                r1 = np.random.rand(self.problem_dim)
                r2 = np.random.rand(self.problem_dim)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta_pos - self.positions[i])
                X2 = self.beta_pos - A2 * D_beta

                r1 = np.random.rand(self.problem_dim)
                r2 = np.random.rand(self.problem_dim)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta_pos - self.positions[i])
                X3 = self.delta_pos - A3 * D_delta

                self.positions[i] = (X1 + X2 + X3) / 3
                self.positions[i] = np.clip(self.positions[i], [b[0] for b in self.bounds], [b[1] for b in self.bounds])

            self.history.append(self.best_fitness)
            print(f"Iteration {iter+1}/{max_iter}, Best Fitness: {self.best_fitness:.6f}")

        print(f"GWO Optimization Complete! Time: {time.time() - start_time:.2f}s")

    def get_best_solution(self) -> np.ndarray:
        return self.best_position

    def get_convergence_history(self):
        return self.history