"""
K-fold CUBV (Cross Upper Bound Validation) Package

This package contains implementations for statistically rigorous cross-validation
methods based on concentration inequalities and PAC-Bayesian theory.

Based on: "Is K-fold cross validation the best model selection method for Machine Learning?"
arXiv:2401.16407

Classes:
--------
KFoldCUBV : Main CUBV implementation with McDiarmid and PAC-Bayesian bounds
ExperimentalEvaluation : Tools for reproducing experiments from the paper

Functions:
----------
plot_power_curves : Plot power analysis results

Example:
--------
from validation.kfold_cubv import KFoldCUBV, ExperimentalEvaluation

# Generate synthetic data
X, y = ExperimentalEvaluation.generate_gaussian_data(100, 10, cohen_d=0.5)

# Run CUBV test
cubv = KFoldCUBV()
result = cubv.cubv_test(X, y)
print(f"Significant: {result['significant']}")
"""

from .kfold_cubv import KFoldCUBV, ExperimentalEvaluation, plot_power_curves

__all__ = ['KFoldCUBV', 'ExperimentalEvaluation', 'plot_power_curves']
