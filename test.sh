#!/bin/bash
echo "Running tests..."
echo "Testing particle.py..."
pytest sedo/tests/test_particle.py -v
echo "Testing optimizer.py..."
pytest sedo/tests/test_optimizer.py -v
echo "Testing utils.py..."
pytest sedo/tests/test_utils.py -v
echo "Testing search.py..."
pytest sedo/tests/test_search.py -v
echo "Test completed."