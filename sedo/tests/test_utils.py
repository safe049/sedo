import os
import numpy as np
from sedo.utils import save_checkpoint, load_checkpoint, plot_convergence

def test_save_load_checkpoint(tmpdir):
    class DummyOptimizer:
        particles = [1, 2, 3]
        global_best_pos = np.array([1, 2, 3])
        global_best_fit = 0.1
        history = [{'iteration': 0, 'best_fitness': 0.1}]
        diversity_history = [0.5]
        temperature = 1.0
    path = tmpdir.join("checkpoint.pkl")
    save_checkpoint(DummyOptimizer(), str(path))
    loaded = load_checkpoint(str(path))
    assert loaded['global_best_fit'] == 0.1
    assert len(loaded['particles']) == 3

def test_plot_convergence():
    from matplotlib import pyplot as plt
    history = [{'best_fitness': i*0.1} for i in range(10)]
    plot_convergence(history)
    plt.close()