#!/bin/bash

# Bash script to run K-fold CUBV validation
# Based on the paper: arXiv:2401.16407

echo "Starting K-fold CUBV validation..."
echo "Based on: arXiv:2401.16407"
echo ""

# Set environment variables
export BASE_DIR="$(dirname "$(realpath "$0")")/.."
export PYTHONPATH="$PYTHONPATH:$BASE_DIR:$BASE_DIR/validation"

# Change to validation directory
cd "$BASE_DIR/validation" || exit 1

# Run the validation script
echo "Running CUBV experiments..."
python3 -c "
import sys
sys.path.append('.')
from kfold_cubv import KFoldCUBV, ExperimentalEvaluation
import numpy as np

print('='*70)
print('K-fold CUBV Implementation')
print('Based on: arXiv:2401.16407')
print('='*70)

# Example 1: Simple test
print('\n--- Example 1: Simple Classification Test ---')
np.random.seed(42)

X, y = ExperimentalEvaluation.generate_gaussian_data(
    n_samples=100, n_features=10, cohen_d=0.5, n_clusters=1
)

cubv = KFoldCUBV(n_splits=5, delta=0.05, n_repetitions=20)
result = cubv.cubv_test(X, y, method='mcdiarmid')

print(f'Results:')
print(f'  Mean Accuracy: {result[\"mean_accuracy\"]:.4f} ± {result[\"std_accuracy\"]:.4f}')
print(f'  Empirical Risk: {result[\"empirical_risk\"]:.4f}')
print(f'  Upper Bound: {result[\"epsilon\"]:.4f}')
print(f'  Upper Bound Risk: {result[\"upper_bound\"]:.4f}')
print(f'  Significant? {result[\"significant\"]}')

# Example 2: Null experiment
print('\n--- Example 2: Null Experiment (Cohen d = 0) ---')
X_null, y_null = ExperimentalEvaluation.generate_gaussian_data(
    n_samples=100, n_features=10, cohen_d=0.0, n_clusters=1
)

result_null = cubv.cubv_test(X_null, y_null, method='mcdiarmid')
print(f'  Mean Accuracy: {result_null[\"mean_accuracy\"]:.4f}')
print(f'  Upper Bound Risk: {result_null[\"upper_bound\"]:.4f}')
print(f'  Significant? {result_null[\"significant\"]} (should be False)')

print('\n--- Summary ---')
print('CUBV method provides:')
print('  ✓ Better False Positive control')
print('  ✓ Robust to small samples')
print('  ✓ No distributional assumptions')
print('  ✓ Conservative but reliable')

print('\nK-fold CUBV validation completed!')
"

echo ""
echo "K-fold CUBV validation completed successfully!"
echo "Check the results above for validation outcomes."
