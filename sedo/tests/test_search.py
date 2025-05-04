import numpy as np
from sedo.search import SEDSearchCV

def dummy_func(x):
    return sum((xi - 1)**2 for xi in x)

def test_sedsearchcv_finds_good_params():
    param_space = {
        'x0': [-5, 5],
        'x1': [-5, 5],
        'x2': [-5, 5]
    }
    searcher = SEDSearchCV(dummy_func, param_space, n_particles=10, max_iter=10)
    searcher.fit()
    assert len(searcher.best_params_) == 3
    assert all(v >= -5 and v <= 5 for v in searcher.best_params_.values())
    assert searcher.best_score_ < 1e-1