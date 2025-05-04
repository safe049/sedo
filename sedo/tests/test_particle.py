import numpy as np
from sedo.particle import QuantumParticle

def test_particle_initialization():
    p = QuantumParticle(problem_dim=5, discrete_dims=[0])
    assert p.cultural_dimension.shape == (6,)
    assert isinstance(p.entropy_phase, complex)
    assert isinstance(p.positive_entropy, float)
    assert isinstance(p.negative_entropy, float)

def test_set_position_with_bounds():
    p = QuantumParticle(problem_dim=3, discrete_dims=[1])
    bounds = [(-1, 1), (-2, 2), (0, 5)]
    p.set_position(np.array([2.0, -3.0, 6.0]), bounds)
    assert np.allclose(p.position, [1.0, -2.0, 5.0])

def test_discrete_variable_rounding():
    p = QuantumParticle(problem_dim=2, discrete_dims=[1])
    bounds = [(0, 5), (0, 10)]
    p.set_position(np.array([2.3, 4.7]), bounds)
    assert p.position[1] == 5