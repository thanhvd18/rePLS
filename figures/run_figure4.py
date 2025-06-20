import os
import sys
import json


from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from icecream import ic

import figures.figure4 as fig4
import figures.figure1 as fig1
from cross_validator import CrossValidator


def plot_fig_4d(show_plot=False):
    np.random.seed(0)
    data = pd.read_csv('figure4/figures/mean_P.csv')
    data = data.iloc[:, 1:].values

    corr = np.corrcoef(data, rowvar=False)
    fig, ax = plt.subplots(figsize=(4, 4))

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            size = abs(corr[i, j]) * 1200
            color = plt.cm.bwr_r((corr[i, j] + 1) / 2)  # Normalize to [0, 1] for color mapping
            ax.scatter(j, i, s=size, c=[color], alpha=1, edgecolors=color)
            if abs(corr[i, j]) < 0.2:
                ax.scatter(j, i, s=70, c='black', marker='x', linewidths=1.5)

    ax.set_xticks(np.arange(-0.5, corr.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, corr.shape[0]), minor=True)

    ax.grid(which='minor', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    ax.set_ylim(-0.5, corr.shape[0] - 0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    os.makedirs("figure4/4d", exist_ok=True)
    plt.savefig('figure4/4d/fig4d.svg')
    # save matrix to csv
    pd.DataFrame(corr).to_csv('figure4/4d/fig4d.csv', index=False)
    if show_plot:
        plt.show()

def plot_fig_4a(cache=True, show_plot=False):
    print("Running")
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    n_components = 5
    n_splits = 10
    n_repeats = 20
    random_state = 1
    out_dir = './figure4/4a'
    csv_dir = os.path.join(out_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)

    cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                        stratified=True, random_state=random_state)

    if not cache or not os.path.exists(os.path.join(csv_dir, 'mean_P.csv'))\
            or not os.path.exists(os.path.join(csv_dir, 'mean_alpha.csv'))\
            or not os.path.exists(os.path.join(csv_dir, 'mean_Q.csv')):
        mean_P, mean_alpha, mean_Q = fig4.utils.k_fold_prediction(df, cv, csv_dir,n_components,random_state,n_splits)

        mean_P_df = pd.DataFrame(mean_P, columns=[f"P{i+1}" for i in range(mean_P.shape[1])])
        mean_P_df.to_csv(os.path.join(csv_dir, 'mean_P.csv'), index=False)

        mean_alpha_df = pd.DataFrame(mean_alpha, columns=[f"alpha{i+1}" for i in range(mean_alpha.shape[1])])
        mean_alpha_df.to_csv(os.path.join(csv_dir, 'mean_alpha.csv'), index=False)

        mean_Q_df = pd.DataFrame(mean_Q, columns=[f"Q{i+1}" for i in range(mean_Q.shape[1])])
        mean_Q_df.to_csv(os.path.join(csv_dir, 'mean_Q.csv'), index=False)
    else:
        print("Loading from cache")
        mean_P = pd.read_csv(os.path.join(csv_dir, 'mean_P.csv')).values
        mean_alpha = pd.read_csv(os.path.join(csv_dir, 'mean_alpha.csv')).values
        mean_Q = pd.read_csv(os.path.join(csv_dir, 'mean_Q.csv')).values

    return mean_P,mean_alpha

def plot_fig_4_PQ():
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    n_components = 5
    n_splits = 10
    n_repeats = 20
    random_state = 1
    out_dir = './figure4/results'
    cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                        stratified=True, random_state=random_state)

    mean_P, mean_alpha = fig4.utils.k_fold_prediction(df, cv, out_dir,n_components,random_state,n_splits)
    #save mean_P to mat
    from scipy import io
    io.savemat('mean_P.mat', {'P': mean_P})
    return mean_P,mean_alpha

def plot_fig_4e():
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    n_components = 5
    n_splits = 10
    n_repeats = 1
    random_state = 1
    out_dir = './figure4/results'
    cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                        stratified=True, random_state=random_state)

    result_df = fig4.utils.sample_size_change_prediction(df,cv, out_dir,n_components,random_state,n_splits)

    sns.lineplot(data=result_df, x="perc", y="r", hue="method", marker="o")
    os.makedirs("figure4/4e", exist_ok=True)
    plt.savefig('figure4/4e/figure4e.svg')
    plt.show()
    result_df.to_csv('figure4/4e/figure4e.csv', index=False)
    return result_df

def plot_fig_4f():
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    n_components = 5
    n_splits = 10
    n_repeats = 1
    random_state = 1
    out_dir = './figure4/results'
    cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                        stratified=True, random_state=random_state)
    result_df = fig4.utils.number_of_outcomes_prediction(df, cv, out_dir, n_components, random_state, n_splits)

    df = result_df.groupby(["n.outcome"])[["r", "mean_MSE"]].mean()

    fig, ax1 = plt.subplots(figsize=(5, 5))
    color = 'tab:red'
    ax1.set_xlabel('n.outcome')
    ax1.set_ylabel('r', color=color)
    sns.lineplot(data=df, x="n.outcome", y="r", ax=ax1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # set lim 0.576 -> 0.585
    ax1.set_ylim(0.576, 0.585)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('mean_MSE', color=color)
    sns.lineplot(data=df, x="n.outcome", y="mean_MSE", ax=ax2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.667, 0.679)
    plt.show()
    os.makedirs("figure4/4f", exist_ok=True)
    plt.savefig('figure4/4f/figure4f.svg')
    plt.show()
    #save data for plotting using other tools
    result_df.to_csv('figure4/4f/figure4f.csv', index=False)
    return result_df

def plot_fig_4a_supplementary():
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)
    df, selected_subjects, labels = fig1.utils.categorize_disease_group(df)

    n_components = 5
    n_repeats = 1
    random_state = 1
    out_dir = './figure4/results/supplementary'
    results = []
    for test_size in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

        if test_size == 0:
            n_splits = len(df['SubjectID'].unique())
            cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                                stratified=False, random_state=random_state)
            mean_P, scores = fig4.utils.k_fold_prediction(df, cv, out_dir,n_components,random_state,n_splits=n_splits)
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(
                out_dir, f"mean_P_test_size{test_size}.mat"
            )
        else:
            sample_train_df, sample_test_df, N_subject_train = fig4.utils.stratified_train_test_df_split(df, test_size,random_state)
            print(f"N_subject_train = {N_subject_train}, perform LOOCV")
            print(f"training size = { len(df['SubjectID'].unique())}")
            n_splits = len(sample_train_df['SubjectID'].unique())
            cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                                stratified=False, random_state=random_state)
            mean_P, scores = fig4.utils.k_fold_prediction(sample_train_df, cv, out_dir, n_components, random_state,
                                                              n_splits=n_splits)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(
            out_dir, f"mean_P_test_size{test_size}.mat"
        )

        scipy.io.savemat(path, {'P': mean_P})
        if isinstance(scores, dict):
            scores_row = scores.copy()
        else:
            scores_row = scores[0] if isinstance(scores, list) and len(scores) > 0 else {}
        scores_row['test_size'] = test_size
        results.append(scores_row)

    scores_df = pd.DataFrame(results)
    scores_csv_path = os.path.join(out_dir, "scores_by_test_size.csv")
    scores_df.to_csv(scores_csv_path, index=False)

def figure_4b():
    # Load data from CSV
    P = pd.read_csv('figure4/4b/mean_P_filter.csv').values
    P = P[2:, :]

    # Load parcellation data
    network_name7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

    # Create DataFrame with network labels
    parcellation_df = pd.read_csv(os.path.join(os.getcwd(), "..",'data/7network_sort_by_index.csv'))
    df = pd.DataFrame(data=np.c_[P, parcellation_df.network.values],
                      columns=[f"P{i + 1}" for i in range(P.shape[1])] + ['network'])
    # Group data by network and sum absolute values
    df_group = df.abs().groupby(["network"]).sum()

    sankey_data = []

    for network, row in df_group.iterrows():
        for col, value in row.items():
                sankey_data.append({
                    "source": network_name7[int(network)],
                    "target": col,
                    "value": float(value)
                })

    # Convert to JSON format
    sankey_json = json.dumps(sankey_data, indent=2)

    # Sankey chart configuration
    option = {
        "series": {
            "type": "sankey",
            "layout": "none",
            "emphasis": {
                "focus": "adjacency"
            },
            "lineStyle": {
                "color": "source",
                "curveness": 0.5
            },
            "data": [
                {
                    "name": "P1",
                    "itemStyle": {
                        "color": "#808080"
                    },
                    "localX": 1,
                    "localY": 0
                },
                {
                    "name": "P2",
                    "itemStyle": {
                        "color": "#808080"
                    },
                    "localX": 1,
                    "localY": 0.23
                },
                {
                    "name": "P3",
                    "itemStyle": {
                        "color": "#808080"
                    },
                    "localX": 1,
                    "localY": 0.51
                },
                {
                    "name": "P4",
                    "itemStyle": {
                        "color": "#808080"
                    },
                    "localX": 1,
                    "localY": 0.73
                },
                {
                    "name": "P5",
                    "itemStyle": {
                        "color": "#808080"
                    },
                    "localX": 1,
                    "localY": 0.79
                },
                {
                    "name": "Vis",
                    "itemStyle": {
                        "color": "#660872"
                    }, "localX": 0,
                    "localY": 0,
                },
                {
                    "name": "SomMot",
                    "itemStyle": {
                        "color": "#5DADE2"
                    }, "localX": 0,
                    "localY": 0.13,
                },
                {
                    "name": "DorsAttn",
                    "itemStyle": {
                        "color": "#1E8449"
                    }, "localX": 0,
                    "localY": 0.3,
                },
                {
                    "name": "SalVentAttn",
                    "itemStyle": {
                        "color": "#E91E63"
                    }, "localX": 0,
                    "localY": 0.43,
                },
                {
                    "name": "Limbic",
                    "itemStyle": {
                        "color": "#6E2C00"
                    }, "localX": 0,
                    "localY": 0.56,
                },
                {
                    "name": "Cont",
                    "itemStyle": {
                        "color": "#E67E22"
                    }, "localX": 0,
                    "localY": 0.64,
                },

                {
                    "name": "Default",
                    "itemStyle": {
                        "color": "#C0392B"
                    },
                    "localX": 0,
                    "localY": 0.79,
                }
            ],
            "links": sankey_data
        }
    };

    print(json.dumps(option, indent=2))
    # "https://echarts.apache.org/examples/en/editor.html?c=sankey-simple"
    # write to json
    with open('figure4/4b/figure4b.json', 'w') as f:
        json.dump(option, f)
    return option

if __name__ == '__main__':
    # plot_fig_4a(cache=False)
    # result = figure_4b()
    # result_df = plot_fig_4e()
    # result_df = plot_fig_4f()
    # print(result_df)
    # plot_fig_4d()
    plot_fig_4a_supplementary()

