import numpy as np
from sedo.optimizer import SEDOptimizer

def sphere(x):
    return sum(xi**2 for xi in x)

def test_optimizer_init():
    opt = SEDOptimizer(sphere, problem_dim=5, n_particles=10)
    assert len(opt.particles) == 10
    assert opt.global_best_fit == float('inf')

def test_optimize_runs():
    opt = SEDOptimizer(sphere, problem_dim=5, n_particles=10)
    opt.optimize(max_iter=10)
    best = opt.get_best_solution()
    assert len(best) == 5
    assert opt.global_best_fit < 1.5

def test_parallel_execution():
    opt = SEDOptimizer(sphere, problem_dim=5, n_particles=10, use_parallel=True)
    opt.optimize(max_iter=10)
    assert opt.global_best_fit < 1.5

def test_restart_mechanism():
    class DummyOpt(SEDOptimizer):
        def _calculate_diversity(self): return 0.0001  # 强制触发重启
    opt = DummyOpt(sphere, problem_dim=5, n_particles=10)
    opt._evaluate_fitness()
    opt._update_diversity()
    opt._restart_if_stagnant()
    assert len(opt.particles) > 0