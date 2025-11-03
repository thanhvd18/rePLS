#!/bin/bash

# Bash script to run K-fold CUBV validation experiments

echo "Running K-fold CUBV validation experiments..."
echo "Based on paper: arXiv:2401.16407"
echo ""

# Set environment variables
export BASE_DIR="$(dirname "$(realpath "$0")")/.."
export PYTHONPATH="$PYTHONPATH:$BASE_DIR:$BASE_DIR/validation"

# Change to validation directory
cd "$BASE_DIR/validation" || exit 1

# Run basic validation tests
echo "Running basic CUBV tests..."
python3 -c "
from kfold_cubv import KFoldCUBV, ExperimentalEvaluation
import numpy as np

print('='*70)
print('K-fold CUBV Validation Tests')
print('='*70)

# Test 1: Medium effect size (should be significant)
print('\n--- Test 1: Medium Effect Size (Cohen d = 0.5) ---')
np.random.seed(42)
X, y = ExperimentalEvaluation.generate_gaussian_data(100, 10, cohen_d=0.5)

cubv = KFoldCUBV(n_splits=5, delta=0.05, n_repetitions=20)
result = cubv.cubv_test(X, y, method='mcdiarmid')

print(f'  Mean Accuracy: {result[\"mean_accuracy\"]:.4f} ± {result[\"std_accuracy\"]:.4f}')
print(f'  Empirical Risk: {result[\"empirical_risk\"]:.4f}')
print(f'  Upper Bound: {result[\"epsilon\"]:.4f}')
print(f'  Upper Bound Risk: {result[\"upper_bound\"]:.4f}')
print(f'  Significant: {result[\"significant\"]}')

# Test 2: Null effect (should NOT be significant)
print('\n--- Test 2: Null Effect (Cohen d = 0.0) ---')
X_null, y_null = ExperimentalEvaluation.generate_gaussian_data(100, 10, cohen_d=0.0)
result_null = cubv.cubv_test(X_null, y_null, method='mcdiarmid')

print(f'  Mean Accuracy: {result_null[\"mean_accuracy\"]:.4f}')
print(f'  Upper Bound Risk: {result_null[\"upper_bound\"]:.4f}')
print(f'  Significant: {result_null[\"significant\"]}')

# Test 3: Small sample robustness
print('\n--- Test 3: Small Sample (N=30) ---')
X_small, y_small = ExperimentalEvaluation.generate_gaussian_data(30, 10, cohen_d=1.0)
result_small = cubv.cubv_test(X_small, y_small, method='mcdiarmid')

print(f'  Mean Accuracy: {result_small[\"mean_accuracy\"]:.4f}')
print(f'  Upper Bound Risk: {result_small[\"upper_bound\"]:.4f}')
print(f'  Significant: {result_small[\"significant\"]}')

print('\n--- Summary ---')
print(f'Medium effect: {\"✓\" if result[\"significant\"] else \"✗\"} significant')
print(f'Null effect: {\"✓\" if not result_null[\"significant\"] else \"✗\"} not significant (correct)')
print(f'Small sample: {\"✓\" if result_small[\"significant\"] else \"✗\"} significant')
print('\nValidation tests completed!')
"

echo ""
echo "Validation experiments completed!"
echo "Results saved in validation/ directory"
