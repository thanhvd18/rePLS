import os
import sys

import seaborn as sns
from icecream import ic
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from umap import UMAP
from sklearn.manifold import TSNE
# Import neural network components for autoencoder
try:
    from sklearn.neural_network import MLPRegressor
    SKLEARN_NN_AVAILABLE = True
except ImportError:
    SKLEARN_NN_AVAILABLE = False

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.getcwd())

from rePLS import rePLS, rePCR, reMLR
import config
import figures.figure3 as fig3
import figures.figure1 as fig1
from cross_validator import CrossValidator


class SklearnAutoencoder:
    """Autoencoder implementation using sklearn's MLPRegressor for neural network functionality."""

    def __init__(self, input_dim, encoding_dim=50, hidden_dims=None):
        if hidden_dims is None:
            # Create hidden layers that bottleneck to encoding_dim
            # For brain imaging data, use larger hidden layers
            hidden_dims = [200, 100, encoding_dim]

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims

        # For sklearn, we'll use a single hidden layer with encoding_dim neurons
        # This acts as a simple autoencoder
        self.autoencoder = None
        self.encoder = None

    def _create_autoencoder_architecture(self):
        """Create the autoencoder architecture using MLPRegressor."""
        if not SKLEARN_NN_AVAILABLE:
            raise ImportError("sklearn neural network not available")

        # For autoencoder, we need to predict the input from itself
        # We'll use a neural network that compresses to encoding_dim and back
        hidden_layer_sizes = tuple(self.hidden_dims)

        # Create autoencoder (input -> encoded -> reconstructed input)
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1
        )

    def fit(self, X, epochs=None, batch_size=None, learning_rate=None, validation_split=0.1):
        """Train the autoencoder."""

        self._create_autoencoder_architecture()

        # For autoencoder training, we predict X from X
        self.autoencoder.fit(X, X)
        return self

    def encode(self, X):
        """Extract encoded features using the hidden layer activations."""

        # For sklearn MLPRegressor, we need a different approach to extract encoded features
        # We'll create a simple linear transformation based on the first layer weights
        if hasattr(self.autoencoder, 'coefs_') and len(self.autoencoder.coefs_) > 0:
            # Get the first layer weights (input to first hidden layer)
            W1 = self.autoencoder.coefs_[0]  # Shape: (input_dim, hidden_dim)

            # Apply the first layer transformation
            encoded = X @ W1

            # If there are more layers, take only up to encoding_dim
            if encoded.shape[1] > self.encoding_dim:
                encoded = encoded[:, :self.encoding_dim]

            return encoded
    

    def fit_transform(self, X, epochs=None, batch_size=None, learning_rate=None, validation_split=0.1):
        """Train autoencoder and return encoded features."""
        self.fit(X, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, validation_split=validation_split)
        return self.encode(X)


def plot_fig_3a(out_dir='./figure3/3a',random_state=1, show_plot=False):
    out_dir = f"{out_dir}_{random_state}"
    data_path = os.path.join(os.path.dirname(__file__), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    n_components = 5
    n_splits = 10
    n_repeats = 1
    os.makedirs(out_dir, exist_ok=True)
    method = 'rePLS'  # ["rePLS", "PLS", "PCR", "rePCR", "LR", "reMLR"]
    cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                        stratified=False, random_state=random_state)

    stat_df, predict_result_df, combine_stat_df = fig3.utils.k_fold_prediction(df, cv, out_dir,method,n_components,random_state,n_splits)
    # predict_result_df: data frame Nx 8 outcomes and their prediction + diagnostic status
    # predict_result_df:   ['outcome0', 'outcome1', 'outcome2', 'outcome3', 'outcome4', 'outcome5', 'outcome6', 'outcome7', 'outcome0_pred', 'outcome1_pred', 'outcome2_pred', 'outcome3_pred', 'outcome4_pred', 'outcome5_pred', 'outcome6_pred', 'outcome7_pred', 'DX_encode', 'idx']
    # combine_stat_df:  ['outcome', 'r', 'MSE', 'p_value']

    # plot scatter plot for each outcome and its prediction
    for i in range(len(config.outcomes)):
        plt.figure(figsize=(5, 5))
        sns.scatterplot(x=predict_result_df[f'outcome{i}'], y=predict_result_df[f'outcome{i}_pred'],
                        hue=predict_result_df['DX_encode'], palette='viridis')
        sns.regplot(x=predict_result_df[f'outcome{i}'], y=predict_result_df[f'outcome{i}_pred'], scatter=False,
                    color='red')

        # determine min max and set equal lim for x and y
        min_val = min(predict_result_df[f'outcome{i}'].min(), predict_result_df[f'outcome{i}_pred'].min())
        max_val = max(predict_result_df[f'outcome{i}'].max(), predict_result_df[f'outcome{i}_pred'].max())
        if i == 7:
            min_val = max(min_val, -50)
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        plt.xlabel(f'Observed value')
        plt.ylabel(f'Predicted value')
        plt.title(
            f'{config.outcomes[i]}, r={combine_stat_df.loc[i, "r"]:.4f}, P={combine_stat_df.loc[i, "p_value"]:.0E}')
        #save figure to svg
        plt.savefig(f'{out_dir}/{config.outcomes[i]}.svg')
        if show_plot:
            plt.show()
        else:
            plt.close()
    return combine_stat_df

def plot_fig_3b(out_dir='./figure3/3b',random_state=1, show_plot=False):
    data_path = os.path.join(os.path.dirname(__file__), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    n_components = 5
    n_splits = 10
    n_repeats = 1

    methods = [ 'rePLS', 'rePCR', 'reMLR']  # ["rePLS", "PLS", "PCR", "rePCR", "LR", "reMLR"]
    cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                        stratified=False, random_state=random_state)
    combine_method_df = pd.DataFrame()
    combine_method_p_value_df = pd.DataFrame()
    combine_method_df["outcome"] = config.outcomes
    combine_method_p_value_df["outcome"] = config.outcomes
    for method in methods:
        if method == 'rePLS' or method == 'PLS':
            n_components = 5
        elif method == 'rePCR' or method == 'PCR':
            n_components = 20
        elif method == 'reMLR':
            n_components = 0
        stat_df, predict_result_df, combine_stat_df = fig3.utils.k_fold_prediction(df, cv, out_dir,
            method, n_components=n_components, random_state=random_state, n_splits=n_splits)
        combine_method_df[method] = combine_stat_df['r']

    df_melted = combine_method_df.melt(id_vars="outcome", var_name="Group", value_name="Value")
    df_melted = df_melted.sort_values(by="Value")

    plt.figure(figsize=(8, 5))
    sns.barplot(x="outcome", y="Value", hue="Group", data=df_melted)

    # Labels
    plt.title("Grouped Bar Plot with Wide-Format Data")
    plt.xlabel("Category")
    plt.ylabel("Values")
    plt.legend(title="Group")
    #save to svg
    os.makedirs("figure3/3b", exist_ok=True)
    plt.savefig('figure3/3b/figure3b.svg')
    if show_plot:
        plt.show()
    else:
        plt.close()
    return

def plot_fig_3b_data_leakage(out_dir='./figure3/3b_data_leakage', random_state=1, show_plot=False):
    """
    Demonstrate data leakage by using all data for both training and 'prediction'.
    This shows how artificially inflated the correlation coefficients become.
    """
    data_path = os.path.join(os.path.dirname(__file__), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    n_components = 5

    methods = ['rePLS', 'rePCR', 'reMLR', 'PCR', 'LR', 'Autoencoder', 'UMAP', 'TSNE']
    cv = CrossValidator(n_splits=10, n_repeats=1, stratified=False, random_state=random_state)

    # First get proper cross-validation results for comparison (skip autoencoder for simplicity)
    print("Getting proper cross-validation results for comparison...")
    proper_results = {}
    cv_methods = ['rePLS', 'rePCR', 'reMLR', 'PCR', 'LR', 'UMAP', 'TSNE']  # Skip autoencoder in CV for simplicity

    for method in cv_methods:
        if method == 'rePLS' or method == 'PLS':
            n_comp = 5
        elif method == 'rePCR' or method == 'PCR':
            n_comp = 20
        elif method == 'UMAP' or method == 'TSNE':
            n_comp = 3
        elif method == 'reMLR' or method == 'LR':
            n_comp = 0

        stat_df, predict_result_df, combine_stat_df = fig3.utils.k_fold_prediction(
            df, cv, out_dir, method, n_components=n_comp,
            random_state=random_state, n_splits=10
        )
        proper_results[method] = combine_stat_df['r'].values

    # Now do data leakage version
    print("Computing data leakage results...")
    leakage_results = {}

    # Use all subjects for training
    selected_subjects = df['SubjectID'].unique()
    df_all = df[df['SubjectID'].isin(selected_subjects)]

    X_all = np.vstack(df_all['Schaefer_200_7'].apply(eval))
    Y_all = np.array(df_all[config.outcomes])
    Z_all = np.array(df_all[['AGE', 'PTGENDER']], dtype=float)  # confounders are AGE and PTGENDER

    # Normalize
    X_scaler = StandardScaler()
    X_all_scaled = X_scaler.fit_transform(X_all)

    Y_scaler = StandardScaler()
    Y_all_scaled = Y_scaler.fit_transform(Y_all)

    for method in methods:
        print(f"Processing method: {method}")
        if method == 'rePLS' or method == 'PLS':
            n_comp = 5
        elif method == 'rePCR' or method == 'PCR':
            n_comp = 20
        elif method == 'reMLR':
            n_comp = 0

        if method == "LR":
            model = LinearRegression()
            model.fit(X_all_scaled, Y_all_scaled)
            y_pred_scaled = model.predict(X_all_scaled)
        elif method == "reMLR":
            model = reMLR(Z=Z_all)
            model.fit(X_all_scaled, Y_all_scaled)
            y_pred_scaled = model.predict(X_all_scaled, Z=Z_all)
        elif method == "PCR":
            model = make_pipeline(PCA(n_components=n_comp), LinearRegression())
            model.fit(X_all_scaled, Y_all_scaled)
            y_pred_scaled = model.predict(X_all_scaled)
        elif method == "rePCR":
            model = rePCR(n_components=n_comp, Z=Z_all)
            model.fit(X_all_scaled, Y_all_scaled)
            y_pred_scaled = model.predict(X_all_scaled, Z=Z_all)
        elif method == "PLS":
            model = PLSRegression(n_components=n_comp)
            model.fit(X_all_scaled, Y_all_scaled)
            y_pred_scaled = model.predict(X_all_scaled)
        elif method == "rePLS":
            model = rePLS(n_components=n_comp, Z=Z_all)
            model.fit(X_all_scaled, Y_all_scaled)
            y_pred_scaled = model.predict(X_all_scaled, Z=Z_all)
        elif method == "Autoencoder":
            # Train autoencoder on input features
            input_dim = X_all_scaled.shape[1]
            autoencoder = SklearnAutoencoder(input_dim=input_dim, encoding_dim=50)
            X_encoded = autoencoder.fit_transform(X_all_scaled)

            # Use encoded features to predict outcomes
            outcome_model = LinearRegression()
            outcome_model.fit(X_encoded, Y_all_scaled)
            y_pred_scaled = outcome_model.predict(X_encoded)
        elif method == "UMAP":
            model = make_pipeline(UMAP(n_components=n_comp), LinearRegression())
            model.fit(X_all_scaled, Y_all_scaled)
            y_pred_scaled = model.predict(X_all_scaled)
        elif method == "TSNE":
            model = make_pipeline(TSNE(n_components=n_comp), LinearRegression())
            model.fit(X_all_scaled, Y_all_scaled)
            y_pred_scaled = model.predict(X_all_scaled)


        # Convert predictions back to original scale
        y_pred = Y_scaler.inverse_transform(y_pred_scaled)
        y_true = Y_all  # Original scale

        # Calculate correlations (this is where leakage creates artificially high values)
        leakage_r_values = []
        for i in range(len(config.outcomes)):
            r, p = pearsonr(y_true[:, i], y_pred[:, i])
            leakage_r_values.append(r)

        leakage_results[method] = leakage_r_values

    # Create comparison dataframe
    comparison_df = pd.DataFrame()
    comparison_df["outcome"] = config.outcomes

    for method in cv_methods:  # Use cv_methods for proper results
        comparison_df[f"{method}_proper"] = proper_results[method]
        comparison_df[f"{method}_leakage"] = leakage_results[method]

        # Calculate inflation factor
        comparison_df[f"{method}_inflation"] = [
            l / p if p != 0 else float('inf')
            for l, p in zip(leakage_results[method], proper_results[method])
        ]

    # Handle autoencoder separately (no proper CV comparison)
    if 'Autoencoder' in leakage_results:
        comparison_df["Autoencoder_leakage"] = leakage_results['Autoencoder']
        # For autoencoder, we can't calculate inflation factor vs proper CV

    if 'UMAP' in leakage_results:
        comparison_df["UMAP_leakage"] = leakage_results['UMAP']
    if 'TSNE' in leakage_results:
        comparison_df["TSNE_leakage"] = leakage_results['TSNE']

        # Create visualization
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(4*n_methods, 6))

    methods_plot = ['rePLS', 'rePCR', 'reMLR', 'PCR', 'LR', 'UMAP', 'TSNE', 'Autoencoder']
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'purple', 'orange', 'teal', 'brown', 'red']

    for idx, method in enumerate(methods_plot):
        ax = axes[idx]

        outcomes_idx = range(len(config.outcomes))
        leakage_vals = leakage_results.get(method)
        if leakage_vals is None:
            ax.axis('off')
            continue

        if method == 'Autoencoder':
            # Autoencoder only has leakage results
            ax.bar(outcomes_idx, leakage_vals, width=0.8,
                    label='Data Leakage', color=colors[idx], alpha=1.0)
            ax.set_title(f'{method}: Data Leakage Only')
        else:
            # Other methods have both proper CV and leakage results
            proper_vals = proper_results.get(method)
            if proper_vals is not None:
                ax.bar([i - 0.2 for i in outcomes_idx], proper_vals, width=0.4,
                        label='Proper CV', color=colors[idx], alpha=0.7)
            ax.bar([i + 0.2 for i in outcomes_idx], leakage_vals, width=0.4,
                    label='Data Leakage', color=colors[idx], alpha=1.0)
            ax.set_title(f'{method}: Proper CV vs Data Leakage')

        ax.set_xlabel('Outcomes')
        ax.set_ylabel('Correlation (r)')
        ax.set_xticks(outcomes_idx)
        ax.set_xticklabels(config.outcomes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.3, 0.8)
    plt.tight_layout()

    # Save results
    os.makedirs(out_dir, exist_ok=True)
    comparison_df.to_csv(f'{out_dir}/data_leakage_comparison.csv', index=False)
    plt.savefig(f'{out_dir}/data_leakage_comparison.svg', dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Results saved to {out_dir}")
    print("\nComparison of correlation coefficients:")
    print(comparison_df.round(4))

    return comparison_df

def plot_fig_3b_supplementary():
    data_path = os.path.join(os.path.dirname(__file__), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    n_components = 5
    n_splits = 10
    n_repeats = 1
    random_state = 1
    out_dir = './figure3/results'

    result_df = pd.DataFrame(columns=["method", 'outcome', 'r', 'MSE', 'p_value'])
    for i,method in enumerate(["rePLS", "PLS", "PCR", "rePCR", "LR", "reMLR"]):
    # method = 'rePCR'  # ["rePLS", "PLS", "PCR", "rePCR", "LR", "reMLR"]
        if method == 'rePLS' or method == 'PLS':
            n_components = 5
        elif method == 'rePCR' or method == 'PCR':
            n_components = 20

        cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                            stratified=False, random_state=random_state)

        stat_df, predict_result_df, combine_stat_df = fig3.utils.k_fold_prediction(df, cv, out_dir, method, n_components=n_components,
                                                                                   random_state=random_state, n_splits=n_splits)
        combine_stat_df['method'] = method
        if i == 0:
            result_df = combine_stat_df
        else:
            result_df = pd.concat([result_df, combine_stat_df],axis=0)
    #ReMLR
    # method = 'reMLR'  # ["rePLS", "PLS", "PCR", "rePCR", "LR", "reMLR"]
    # cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
    #                     stratified=False, random_state=random_state)
    #
    # stat_df, predict_result_df, combine_stat_df = fig3.utils.k_fold_prediction(df, cv, out_dir, method, n_components,
    #
    #
    #
    #                                                                            random_state, n_splits)
    path = os.path.join(out_dir, "compare_methods_fig3.csv")
    result_df.to_csv(path, index=False)


    print("Done")
    return


if __name__ == '__main__':
    # plot_fig_3a()
    # plot_fig_3b()
    # plot_fig_3b_supplementary()
    plot_fig_3b_data_leakage(show_plot=True)



