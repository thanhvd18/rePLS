import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import Tuple, Dict, List, Optional, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from icecream import ic

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
# import umap  # Commented out to avoid import errors if not installed

from figures import figure4 as fig4
from figures import figure1 as fig1
from cross_validator import CrossValidator
from figures.figure1 import categorize_disease_group, get_input_output_confounder
from sklearn.preprocessing import StandardScaler

# add src to path to import rePLS family
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from rePLS import rePLS, rePCR, reMLR

from validation.kfold_cubv import KFoldCUBV
class UpperBoundSignificanceTest:
    """
    Implements the upper-bound significance test from the paper.
    Based on PAC-Bayes bounds to determine if classification is better than chance.
    """

    def __init__(self, delta: float = 0.05):
        """
        Args:
            delta: Confidence level (default 0.05 for 95% confidence)
        """
        self.delta = delta

    def compute_pac_bayes_bound(self,
                                empirical_error: float,
                                n_samples: int,
                                delta: float = None) -> float:
        """
        Compute PAC-Bayes upper bound on true error.

        Based on the CUBV framework from the paper (Equation 1-2):
        R(f) ≤ R_CV(f) + Ψ(n, δ)
        where Ψ(n, δ) = sqrt(C * log(1/δ) / (2n))

        Args:
            empirical_error: Cross-validated error rate
            n_samples: Number of samples
            delta: Confidence parameter (if None, use self.delta)

        Returns:
            upper_bound_error: Upper bound on true error
        """
        if delta is None:
            delta = self.delta

        # Concentration term (simplified - paper uses more complex C based on VC dimension)
        C = 1.0  # Simplified constant
        psi = np.sqrt(C * np.log(1.0 / delta) / (2.0 * n_samples))

        # Upper bound on error
        upper_bound_error = empirical_error + psi

        return upper_bound_error

    def test_significance(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         cv_folds: int = 5) -> Dict:
        """
        Test if classification is significantly better than chance.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Class labels (n_samples,)
            cv_folds: Number of CV folds

        Returns:
            Dictionary with empirical_error, corrected_error, is_significant
        """
        n_samples = len(y)

        # Perform stratified cross-validation
        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Get accuracies
        accuracies = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        empirical_accuracy = accuracies.mean()
        empirical_error = 1.0 - empirical_accuracy

        # Apply upper bound correction
        corrected_error = self.compute_pac_bayes_bound(empirical_error, n_samples)
        corrected_accuracy = 1.0 - corrected_error

        # Significant if corrected accuracy > 0.5 (better than chance)
        is_significant = corrected_accuracy > 0.5

        return {
            'empirical_accuracy': empirical_accuracy,
            'empirical_error': empirical_error,
            'corrected_error': corrected_error,
            'corrected_accuracy': corrected_accuracy,
            'is_significant': is_significant,
            'cv_std': accuracies.std()
        }



 

print("Running")
data_path = os.path.join(os.path.dirname(__file__), "..", 'data/ALL_3.csv')
df = fig1.utils.preprocess_df(data_path)

n_components = 5
n_splits = 10
n_repeats = 1
random_state = 1
out_dir = 'results'
csv_dir = os.path.join(out_dir, 'csv')
os.makedirs(csv_dir, exist_ok=True)

cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                    stratified=True, random_state=random_state)


# mean_P, mean_alpha, mean_Q = fig4.utils.k_fold_prediction(df, cv, csv_dir,n_components,random_state,n_splits)
df, selected_subjects, labels = categorize_disease_group(df)
X, Y, Z = get_input_output_confounder(df)
# clf = LogisticRegression(max_iter=1000, random_state=42)
# accuracies = cross_val_score(clf, X[:, 0], y, cv=cv, scoring='accuracy')
selected_subjects = df['SubjectID'].unique()
outcomes = Y.columns.values
confounders = Z.columns.values
n_outcomes = Y.shape[1]

stat_df = pd.DataFrame(columns=["fold", "r", "MSE", "p_value"])
predict_result_df = pd.DataFrame(columns=["outcome" + str(i) for i in range(
    n_outcomes)] + ["outcome" + str(i) + "_pred" for i in range(n_outcomes)] + ["DX_encode"])
df['DX_encode'] = df['DX'].map({'CN': 0, 'MCI': 1, 'AD': 2})

Ps = []
Qs = []
alphas = []

# Initialize result collections
all_correlations = []
all_significant_differences = []
all_latent_differences = []
all_table4_results = []
all_significant_counts = []

# Initialize variables to avoid NameError
corr_df = None
significant_differences = []
latent_differences = []

for fold, (train_index, test_index) in enumerate(cv.get_splits(selected_subjects,labels)):
    print(f"Processing fold {fold}")
    train_subjects = np.array(selected_subjects)[train_index]
    test_subjects = np.array(selected_subjects)[test_index]

    # Reset variables for each fold
    corr_df = None
    significant_differences = []
    latent_differences = []

    df_train = df[df['SubjectID'].isin(train_subjects)]
    df_test = df[df['SubjectID'].isin(test_subjects)]

    X_train = np.vstack(df_train['Schaefer_200_7'].apply(eval))
    Y_train = np.array(df_train[outcomes])
    Z_train = np.array(df_train[confounders], dtype=float)

    X_test = np.vstack(df_test['Schaefer_200_7'].apply(eval))
    Y_test = np.array(df_test[outcomes])
    Y_test_ = np.copy(Y_test)
    Z_test = np.array(df_test[confounders], dtype=float)

    X_train, X_test = X_train[:, :], X_test[:, :]

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    # Y_scaler = StandardScaler()
    # Y_train = Y_scaler.fit_transform(Y_train)
    # Y_test = Y_scaler.transform(Y_test)


    model = rePLS(Z=Z_train, n_components=n_components)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test, Z=Z_test)

    # get latent representation
    T_train = model.transform(X_train, Z=Z_train)
    T_test = model.transform(X_test, Z=Z_test)
    method = "rePLS"
    # T_train from PCA
    # pca = PCA(n_components=n_components)
    # pca.fit(X_train)
    # T_train = pca.transform(X_train)
    # T_test = pca.transform(X_test)
    # method = "pca"
    # change to tSNE
    # tsne = TSNE(n_components=3)
    # tsne.fit(X_train)
    # T_train = tsne.fit_transform(X_train)
    # method = "tsne"
    # T_test = tsne.transform(X_test)
    
    # calcualte the corelation between T_train and X_train
    # for i in range(T_train.shape[1]):
    #     region_correlations = []
    #     for j in range(X_train.shape[1]):
    #         corr = np.corrcoef(T_train[:, i], X_train[:, j])
    #         region_correlations.append(corr[0, 1])
    #     print(f"Region {i} correlations: {region_correlations}")
    #     #save region_correlations to csv
    #     pd.DataFrame(region_correlations).to_csv(f"region_correlations_{method}_{i}_{fold}.csv", index=False)

    # Test the test_significance function
    # Create binary labels for AD vs CN (1 for AD, 0 for CN)
    binary_labels = (df_train['DX'] == 'AD').astype(int).values
    # CN
    # combine
    AD_index = np.where(df_train['DX'] == 'AD')[0]
    CN_index = np.where(df_train['DX'] == 'CN')[0]
    MCI_index = np.where(df_train['DX'] == 'MCI')[0]
    AD_CN_index = np.concatenate([AD_index, CN_index])
    Y_label = df_train['DX_encode'].values[AD_CN_index]
    
    cubv = KFoldCUBV(n_splits=5, delta=0.05, n_repetitions=10)
    
    for i in range(T_train.shape[1]):
        significant_count = 0
        significant_pairs = []  
        T_train_i = T_train[:, i]
        for j in range(X_train.shape[1]):
            X_train_j = X_train[:, j]
            X_train_combined = np.column_stack([T_train, X_train])[AD_CN_index,:]
            example_X = np.column_stack([X_train, X_train_j])[AD_CN_index,:]
            result = cubv.cubv_test(example_X,Y_label,method='pac_bayesian')
            print(result['mean_accuracy'])
            if result['significant']:
                significant_count += 1
                print(f"Fold {fold} - component {i} and region {j} is significant: {significant_count}/202-  { result['mean_accuracy']}")
                # print(significant_count)
                significant_pairs.append((i, j))
            else:
                print(f"Fold {fold} - component {i} and region {j} is not significant: {significant_count}/202-  { result['mean_accuracy']}")
        print(f"Fold {fold} - component {i} - Total significant region pairs: {significant_count} out of {X_train.shape[1]}")
        print(f"Fold {fold} - component {i} - Significant region pairs: {significant_pairs}")
                
    #  for i in range(T_train.shape[1]):
    # if result['significant']:
    # # Initialize the tester
    # tester = UpperBoundSignificanceTest(delta=0.05)

    # # Initialize counter for significant regions
    # significant_count = 0
    # significant_pairs = []

    # # Test each latent component and brain region pair
    # for i in range(T_train.shape[1]):
    #     print(f"Processing component {i}")
    #     print(f"Processing {X_train.shape[1]} brain regions")
    #     significant_count = 0
    #     significant_pairs = []
    #     for j in range(X_train.shape[1]):
    #         example_X = np.column_stack([T_train[:, i], X_train[:, j]])
    #         result = tester.test_significance(example_X, binary_labels, cv_folds=5)
    #         # scatter plot the example_X
    #         plt.figure(figsize=(10, 10))
    #         plt.scatter(example_X[:, 0], example_X[:, 1], c=binary_labels, cmap='viridis')
    #         plt.savefig(f"scatter_plot_{i}_{j}.png")
    #         plt.close()
    #         if result['is_significant']:
    #             significant_count += 1
    #             significant_pairs.append((i, j))
    #             print(f"Fold {fold} - test_significance results for component {i} and region {j}:")
    #             print(f"  Empirical Accuracy: {result['empirical_accuracy']:.3f}")
    #             print(f"  Corrected Accuracy: {result['corrected_accuracy']:.3f}")
    #             print(f"  Is Significant: {result['is_significant']}")
    #     print(f"Total significant region pairs: {significant_count} out of {X_train.shape[1]}")
    # print(f"Fold {fold} - Total significant region pairs: {significant_count} out of {X_train.shape[1]}")

    # # Collect significant count for this fold
    # all_significant_counts.append(significant_count)

# After all folds, save summary results
# print("\n" + "="*80)
# print("SAVING SUMMARY RESULTS")
# print("="*80)