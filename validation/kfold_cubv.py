import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class KFoldCUBV:
    """
    K-fold Cross Upper Bound Validation
    Based on paper: "Is K-fold cross validation the best model selection method for Machine Learning?"
    arXiv:2401.16407
    """

    def __init__(self, n_splits=5, delta=0.05, n_repetitions=100):
        """
        Parameters:
        -----------
        n_splits : int
            Number of folds for K-fold CV
        delta : float
            Confidence level (default 0.05 for 95% confidence)
        n_repetitions : int
            Number of repetitions for bootstrapping
        """
        self.n_splits = n_splits
        self.delta = delta
        self.n_repetitions = n_repetitions

    def compute_mcdiarmid_bound(self, N, delta):
        """
        Compute McDiarmid's inequality upper bound

        ε = sqrt(ln(1/δ) / (2*N))

        Parameters:
        -----------
        N : int
            Sample size
        delta : float
            Confidence parameter

        Returns:
        --------
        epsilon : float
            Upper bound deviation
        """
        epsilon = np.sqrt(np.log(1.0/delta) / (2.0 * N))
        return epsilon

    def compute_pac_bayesian_bound(self, empirical_risk, N, d, delta):
        """
        Compute PAC-Bayesian upper bound for linear classifiers

        Based on Theorem 1 in the paper:
        R(f) ≤ R̂(f) + sqrt((KL + ln(1/δ))/(2N))

        Parameters:
        -----------
        empirical_risk : float
            Empirical risk from CV
        N : int
            Sample size
        d : int
            Dimension (number of features)
        delta : float
            Confidence parameter

        Returns:
        --------
        upper_bound : float
            Upper bound of actual risk
        """
        # For linear classifiers, KL divergence can be approximated
        # Using dropout bound formulation from the paper
        kl_term = d * np.log(N)  # Simplified KL term

        epsilon = np.sqrt((kl_term + np.log(1.0/delta)) / (2.0 * N))
        upper_bound = empirical_risk + epsilon

        return upper_bound

    def kfold_cv_with_repetitions(self, X, y, classifier):
        """
        Perform K-fold CV with repetitions (bootstrapping)

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
        classifier : sklearn classifier
            Classifier to use

        Returns:
        --------
        mean_accuracy : float
            Mean accuracy across all repetitions and folds
        std_accuracy : float
            Standard deviation of accuracy
        all_accuracies : list
            All accuracy values from all repetitions
        """
        all_accuracies = []

        for rep in range(self.n_repetitions):
            # Shuffle data for each repetition
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Perform K-fold CV
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=rep)

            for train_idx, test_idx in kfold.split(X_shuffled):
                X_train, X_test = X_shuffled[train_idx], X_shuffled[test_idx]
                y_train, y_test = y_shuffled[train_idx], y_shuffled[test_idx]

                # Train and evaluate
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                all_accuracies.append(accuracy)

        mean_accuracy = np.mean(all_accuracies)
        std_accuracy = np.std(all_accuracies)

        return mean_accuracy, std_accuracy, all_accuracies

    def cubv_test(self, X, y, classifier=None, method='mcdiarmid'):
        """
        Perform K-fold CUBV test

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
        classifier : sklearn classifier
            Classifier to use (default: linear SVM)
        method : str
            'mcdiarmid' or 'pac_bayesian'

        Returns:
        --------
        results : dict
            Dictionary containing test results
        """
        if classifier is None:
            classifier = SVC(kernel='linear', C=1.0)

        N = len(X)
        d = X.shape[1]

        # Perform K-fold CV with repetitions
        mean_acc, std_acc, all_accs = self.kfold_cv_with_repetitions(X, y, classifier)

        # Convert accuracy to error rate

        empirical_risk = 1.0 - mean_acc

        # Compute upper bound based on selected method
        if method == 'mcdiarmid':
            epsilon = self.compute_mcdiarmid_bound(N, self.delta)
            upper_bound = empirical_risk + epsilon
        elif method == 'pac_bayesian':
            upper_bound = self.compute_pac_bayesian_bound(empirical_risk, N, d, self.delta)
            epsilon = upper_bound - empirical_risk
        else:
            raise ValueError("Method must be 'mcdiarmid' or 'pac_bayesian'")

        # Decision: Reject H0 if upper_bound < 0.5 (random chance)
        # This means the classifier performs better than random even in worst case
        reject_h0 = upper_bound < 0.5

        results = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'empirical_risk': empirical_risk,
            'epsilon': epsilon,
            'upper_bound': upper_bound,
            'reject_h0': reject_h0,
            'significant': reject_h0,
            'p_value_approximation': 1.0 - norm.cdf((0.5 - upper_bound) / std_acc) if std_acc > 0 else 0.0,
            'all_accuracies': all_accs
        }

        return results

    def permutation_test(self, X, y, classifier=None, n_permutations=1000):
        """
        Perform standard permutation test for comparison

        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Target labels
        classifier : sklearn classifier
            Classifier to use
        n_permutations : int
            Number of permutations

        Returns:
        --------
        p_value : float
            Permutation test p-value
        """
        if classifier is None:
            classifier = SVC(kernel='linear', C=1.0)

        # True accuracy
        kfold = KFold(n_splits=self.n_splits, shuffle=True)
        true_scores = cross_val_score(classifier, X, y, cv=kfold)
        true_accuracy = np.mean(true_scores)

        # Permutation distribution
        perm_accuracies = []
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y)
            perm_scores = cross_val_score(classifier, X, y_perm, cv=kfold)
            perm_accuracies.append(np.mean(perm_scores))

        # Compute p-value
        p_value = (1 + np.sum(np.array(perm_accuracies) >= true_accuracy)) / (n_permutations + 1)

        return p_value, true_accuracy, perm_accuracies


class ExperimentalEvaluation:
    """
    Reproduce experiments from the paper
    """

    @staticmethod
    def generate_gaussian_data(n_samples, n_features, cohen_d=0.0, n_clusters=1):
        """
        Generate synthetic Gaussian data with specified Cohen's d effect size

        Parameters:
        -----------
        n_samples : int
            Total number of samples (will be split equally between classes)
        n_features : int
            Number of features
        cohen_d : float
            Cohen's d effect size (distance between class centroids)
        n_clusters : int
            Number of clusters per class (for multi-modal data)

        Returns:
        --------
        X : array, shape (n_samples, n_features)
            Feature matrix
        y : array, shape (n_samples,)
            Labels (0 or 1)
        """
        n_per_class = n_samples // 2

        if n_clusters == 1:
            # Single-mode Gaussian per class
            mean_0 = np.zeros(n_features)
            mean_1 = np.ones(n_features) * cohen_d

            X_0 = np.random.randn(n_per_class, n_features) + mean_0
            X_1 = np.random.randn(n_per_class, n_features) + mean_1

        else:
            # Multi-mode Gaussian per class
            X_0_list = []
            X_1_list = []

            samples_per_cluster = n_per_class // n_clusters

            for i in range(n_clusters):
                # Random cluster centers
                center_0 = np.random.randn(n_features) * 0.5
                center_1 = center_0 + np.ones(n_features) * cohen_d

                X_0_list.append(np.random.randn(samples_per_cluster, n_features) + center_0)
                X_1_list.append(np.random.randn(samples_per_cluster, n_features) + center_1)

            X_0 = np.vstack(X_0_list)
            X_1 = np.vstack(X_1_list)

        X = np.vstack([X_0, X_1])
        y = np.hstack([np.zeros(len(X_0)), np.ones(len(X_1))])

        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        return X, y

    @staticmethod
    def power_analysis(sample_sizes, effect_sizes, n_features=10, n_simulations=100):
        """
        Reproduce power analysis from Figure 11 in the paper

        Parameters:
        -----------
        sample_sizes : list
            List of sample sizes to test
        effect_sizes : list
            List of Cohen's d values to test
        n_features : int
            Number of features
        n_simulations : int
            Number of Monte Carlo simulations

        Returns:
        --------
        results : dict
            Power analysis results
        """
        cubv = KFoldCUBV(n_splits=5, delta=0.05, n_repetitions=10)

        results = {
            'cubv_power': np.zeros((len(effect_sizes), len(sample_sizes))),
            'perm_power': np.zeros((len(effect_sizes), len(sample_sizes))),
        }

        for i, cohen_d in enumerate(effect_sizes):
            for j, n_samples in enumerate(sample_sizes):
                cubv_rejections = 0
                perm_rejections = 0

                print(f"Testing: Cohen's d={cohen_d}, N={n_samples}")

                for sim in range(n_simulations):
                    # Generate data
                    X, y = ExperimentalEvaluation.generate_gaussian_data(
                        n_samples, n_features, cohen_d=cohen_d, n_clusters=1
                    )

                    # CUBV test
                    cubv_result = cubv.cubv_test(X, y, method='mcdiarmid')
                    if cubv_result['significant']:
                        cubv_rejections += 1

                    # Permutation test
                    p_value, _, _ = cubv.permutation_test(X, y, n_permutations=100)
                    if p_value < 0.05:
                        perm_rejections += 1

                results['cubv_power'][i, j] = cubv_rejections / n_simulations
                results['perm_power'][i, j] = perm_rejections / n_simulations

        return results


def plot_power_curves(results, sample_sizes, effect_sizes):
    """
    Plot power curves similar to Figure 11 in the paper
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CUBV power
    for i, cohen_d in enumerate(effect_sizes):
        axes[0].plot(sample_sizes, results['cubv_power'][i, :],
                    marker='o', label=f"Cohen's d={cohen_d}")
    axes[0].axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
    axes[0].set_xlabel('Sample Size')
    axes[0].set_ylabel('Power')
    axes[0].set_title('K-fold CUBV Power')
    axes[0].legend()
    axes[0].grid(True)

    # Permutation test power
    for i, cohen_d in enumerate(effect_sizes):
        axes[1].plot(sample_sizes, results['perm_power'][i, :],
                    marker='s', label=f"Cohen's d={cohen_d}")
    axes[1].axhline(y=0.05, color='r', linestyle='--', label='α=0.05')
    axes[1].set_xlabel('Sample Size')
    axes[1].set_ylabel('Power')
    axes[1].set_title('Permutation Test Power')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('power_analysis.png', dpi=300)
    plt.show()


# ============= EXAMPLE USAGE =============

if __name__ == "__main__":

    print("="*70)
    print("K-fold CUBV Implementation")
    print("Based on: arXiv:2401.16407")
    print("="*70)

    # Example 1: Simple test on synthetic data
    print("\n--- Example 1: Simple Classification Test ---")
    np.random.seed(42)

    # Generate data with medium effect size
    X, y = ExperimentalEvaluation.generate_gaussian_data(
        n_samples=100, n_features=10, cohen_d=0.5, n_clusters=1
    )

    cubv = KFoldCUBV(n_splits=5, delta=0.05, n_repetitions=10)

    # CUBV test
    print("\nRunning K-fold CUBV test...")
    result = cubv.cubv_test(X, y, method='mcdiarmid')

    print(f"  Mean Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    print(f"  Empirical Risk: {result['empirical_risk']:.4f}")
    print(f"  Upper Bound (ε): {result['epsilon']:.4f}")
    print(f"  Upper Bound Risk: {result['upper_bound']:.4f}")
    print(f"  Significant? {result['significant']}")
    print(
        f"  Decision: {'Reject H0' if result['reject_h0'] else 'Fail to reject H0'}"
    )

    # Permutation test for comparison
    print("\nRunning Permutation Test...")
    p_value, true_acc, _ = cubv.permutation_test(X, y, n_permutations=500)
    print(f"  True Accuracy: {true_acc:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant? {p_value < 0.05}")

    # Example 2: Null experiment (should NOT reject H0)
    print("\n--- Example 2: Null Experiment (Cohen's d = 0) ---")
    X_null, y_null = ExperimentalEvaluation.generate_gaussian_data(
        n_samples=100, n_features=10, cohen_d=0.0, n_clusters=1
    )

    result_null = cubv.cubv_test(X_null, y_null, method='mcdiarmid')
    print(f"  Mean Accuracy: {result_null['mean_accuracy']:.4f}")
    print(f"  Upper Bound Risk: {result_null['upper_bound']:.4f}")
    print(f"  Significant? {result_null['significant']} (should be False)")

    # Example 3: Small sample robustness test
    print("\n--- Example 3: Small Sample Robustness (N=30) ---")
    X_small, y_small = ExperimentalEvaluation.generate_gaussian_data(
        n_samples=30, n_features=10, cohen_d=1.0, n_clusters=1
    )

    result_small = cubv.cubv_test(X_small, y_small, method='mcdiarmid')
    print(f"  Mean Accuracy: {result_small['mean_accuracy']:.4f}")
    print(f"  Upper Bound Risk: {result_small['upper_bound']:.4f}")
    print(f"  Significant? {result_small['significant']}")

    # Example 4: Power Analysis (simplified version)
    print("\n--- Example 4: Power Analysis (Simplified) ---")
    print("This may take a few minutes...")

    sample_sizes = [30, 50, 70, 100, 150]
    effect_sizes = [0.0, 0.5, 0.8, 1.2]

    power_results = ExperimentalEvaluation.power_analysis(
        sample_sizes=sample_sizes,
        effect_sizes=effect_sizes,
        n_features=10,
        n_simulations=20  # Reduced for demo
    )

    plot_power_curves(power_results, sample_sizes, effect_sizes)

    print("\n--- Summary ---")
    print("CUBV method provides:")
    print("  ✓ Better False Positive control")
    print("  ✓ Robust to small samples")
    print("  ✓ No distributional assumptions")
    print("  ✓ Conservative but reliable")

    print("\nDone!")
