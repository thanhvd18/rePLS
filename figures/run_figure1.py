import os
import sys
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic
from scipy.spatial.distance import correlation
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from seaborn import color_palette
import seaborn as sns

import config
import figures.figure1 as fig1

def plot_fig_1c()-> Tuple[plt.Axes, plt.Axes, pd.DataFrame, pd.DataFrame]:
    """
        Cortical thickness of 7 networks different between gender.
    """
    save_dir = "figure1/1c"
    csv_dir = os.path.join(save_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    network_df = fig1.utils.mapping_to_7_network(X,
        seven_network_path=os.path.join(os.getcwd(), "..",'data/7network.csv'))
    X.reset_index(drop=True, inplace=True)
    Z.reset_index(drop=True, inplace=True)
    Y.reset_index(drop=True, inplace=True)
    network_df = pd.concat([network_df,X, Z,Y], axis=1)
    network_df['AGE_group'] = network_df.AGE.apply(lambda x: 0 if ( x <= 60) else
        (1 if (x > 60 and x <= 70)
        else (2 if (x > 70 and x <= 80)
        else 3)))

    # PLOT
    network_name7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

    means = []
    df_groups = []
    for g, df_group in network_df.groupby('PTGENDER'):
        df_groups.append(df_group)

    for network in network_name7:
        mean_diff = df_groups[0][network].mean() - df_groups[1][network].mean()
        means.append(mean_diff)

    sorted_means, sorted_categories = zip(*sorted(zip(means, network_name7)))
    x = range(len(sorted_categories))

    fig, ax = plot_bar_chart_1c(network_df, svg=True)
    male_group = X[Z["PTGENDER"] == 1]
    female_group = X[Z["PTGENDER"] == 0]
    t_stat, p_value = ttest_ind(female_group, male_group)


    male_cortical_thickness = male_group.mean().tolist()
    female_cortical_thickness = female_group.mean().tolist()

    male_save_path = f"{csv_dir}/male_cortical_thickness.csv"
    pd.DataFrame(data=male_cortical_thickness, columns=["corical_thickness"]).to_csv(male_save_path, index=False)
    print(f'Saved mean cortical thickness of male to {male_save_path}')
    female_save_path = f"{csv_dir}/female_cortical_thickness.csv"
    pd.DataFrame(data=female_cortical_thickness, columns=["corical_thickness"]).to_csv(female_save_path, index=False)
    print(f'Saved mean cortical thickness of female to {female_save_path}')
    group_comparision_save_path = f"{csv_dir}/t_stat.csv"
    pd.DataFrame(data=t_stat, columns=["corical_thickness"]).to_csv(group_comparision_save_path, index=False)
    print(f'Saved t_stat to {group_comparision_save_path}')

    return ((sorted_means, sorted_categories), male_cortical_thickness, female_cortical_thickness, t_stat)

def plot_fig_1d()-> Tuple[plt.Axes, plt.Axes, pd.DataFrame, pd.DataFrame]:
    save_dir = "figure1/1d"
    csv_dir = os.path.join(save_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    network_df = fig1.utils.mapping_to_7_network(X,
        seven_network_path=os.path.join(os.getcwd(), "..",'data/7network.csv'))
    Z.reset_index(drop=True, inplace=True)
    Y.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)

    network_df = pd.concat([network_df, X, Z, Y], axis=1)
    network_df['AGE_group'] = network_df.AGE.apply(lambda x: 0 if (x <= 60) else
    (1 if (x > 60 and x <= 70)
     else (2 if (x > 70 and x <= 80)
           else 3)))

    fig, ax = plot_fig_1d_2(network_df, svg=True)

    fig, ax = plot_fig_1d_2(network_df, svg=False)

    
    age_group1 = X[network_df["AGE_group"] == 0]
    age_group2 = X[network_df["AGE_group"] == 1]
    age_group3 = X[network_df["AGE_group"] == 2]
    age_group4 = X[network_df["AGE_group"] == 3]

    t_stat1, p_value1 = ttest_ind(age_group1, age_group4)

    group1_cortical_thickness = age_group1.mean().tolist()
    group2_cortical_thickness = age_group2.mean().tolist()
    group3_cortical_thickness = age_group3.mean().tolist()
    group4_cortical_thickness = age_group4.mean().tolist()

    pd.DataFrame(data=group1_cortical_thickness, columns=["corical_thickness"]).to_csv(f"{csv_dir}/group1_cortical_thickness.csv", index=False)
    pd.DataFrame(data=group2_cortical_thickness, columns=["corical_thickness"]).to_csv(f"{csv_dir}/group2_cortical_thickness.csv", index=False)
    pd.DataFrame(data=group3_cortical_thickness, columns=["corical_thickness"]).to_csv(f"{csv_dir}/group3_cortical_thickness.csv", index=False)
    pd.DataFrame(data=group4_cortical_thickness, columns=["corical_thickness"]).to_csv(f"{csv_dir}/group4_cortical_thickness.csv", index=False)
    pd.DataFrame(data=t_stat1, columns=["corical_thickness"]).to_csv(f"{csv_dir}/group1-group4_ttest.csv", index=False)


def plot_fig_1e(show_plot=False)-> Tuple[plt.Axes, plt.Axes, pd.DataFrame, pd.DataFrame]:
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    network_df = fig1.utils.mapping_to_7_network(X,
                                                 seven_network_path=os.path.join(os.getcwd(), "..",
                                                               'data/7network.csv'))
    X_scaler = StandardScaler()
    X = pd.DataFrame(X_scaler.fit_transform(X), columns=X.columns)
    Y_scaler = StandardScaler()
    Y = pd.DataFrame(Y_scaler.fit_transform(Y), columns=Y.columns)

    Z.reset_index(drop=True, inplace=True)
    Y.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)

    network_df = pd.concat([network_df, X, Z, Y], axis=1)
    os.makedirs("figure1/1e/", exist_ok=True)
    network_df.to_csv("figure1/1e/XYZ.csv", index=False)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    x_axis_columns = network_df.loc[:, config.networks]
    y_axis_columns = network_df.loc[:, config.outcomes]

    corr_df = pd.DataFrame(np.corrcoef(y_axis_columns.T, x_axis_columns.T)[:len(config.outcomes),len(config.outcomes):],
                           index=config.outcomes,
                           columns= config.networks)
    plt.figure(figsize=(8, 6))
    sns.set(style="white")

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    ax = sns.heatmap(corr_df, cmap=cmap, annot=True, fmt=".2f",
                     linewidths=0.5, center=0, square=False, cbar_kws={"shrink": 0.5})

    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.title("Full Correlation Matrix: Networks vs Outcomes", fontsize=14)
    corr_df.to_csv("figure1/1e/corr_df.csv", index=False)
    plt.savefig('figure1/1e/figures_1e.svg', format='svg')
    if show_plot:
        plt.show()

    plt.figure(figsize=(6, 6))
    sns.boxplot(data=x_axis_columns)
    plt.xlabel("Outcomes")
    plt.ylabel("Values")
    plt.tight_layout()
    plt.savefig('figure1/1e/figures_1e_1.svg', format='svg')
    if show_plot:
        plt.show()

    # Boxplot for group2
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=y_axis_columns)
    plt.xlabel("Networks")
    plt.ylabel("Values")

    plt.tight_layout()
    plt.savefig('figure1/1e/figures_1e_2.svg', format='svg')
    if show_plot:
        plt.show()

def plot_fig_1f(show_plot=False)-> Tuple[pd.DataFrame, pd.DataFrame]:
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    df_groups = []
    X_group = {}
    for g, df_group in df.groupby('DX'):
        df_groups.append(df_group)
        X, Y, Z = fig1.utils.get_input_output_confounder(df_group)
        X_group[g] = X

    v_min = min([x.mean().min() for x in X_group.values()])
    v_max = max([x.mean().max() for x in X_group.values()])

    save_dir = "figure1/1f"

    csv_folder = os.path.join(save_dir, "csv")
    os.makedirs(csv_folder, exist_ok=True)

    custom_cmap = 'jet_r'
    padding = 30
    for group in ["CN", "MCI", "AD"]:
        post_fix = group
        pd.DataFrame(data=X_group[group].mean().tolist(), columns=["corical_thickness"]).to_csv(
            f"{csv_folder}/{group}_group.csv", index=False)


    for (group1, group2) in [("CN", "MCI"), ("MCI", "AD")]:
        t_stat1, p_value1 = ttest_ind(X_group[group1], X_group[group2])
        post_fix = f"{group1}-{group2}_ttest"
        padding = -55
        pd.DataFrame(data=t_stat1, columns=["corical_thickness"]).to_csv(f"{csv_folder}/{group1}-{group2}_ttest.csv",
                                                                         index=False)

def plot_fig_1d_2(df, svg=True):
    network_name7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

    default = []
    cont = []
    limbic = []
    salvenattn = []
    dorsattn = []
    sommot = []
    vis = []
    for g, df_group in df.groupby('AGE_group'):
        default.append(df_group['Default'].mean())
        cont.append(df_group['Cont'].mean())
        limbic.append(df_group['Limbic'].mean())
        salvenattn.append(df_group['SalVentAttn'].mean())
        dorsattn.append(df_group['DorsAttn'].mean())
        sommot.append(df_group['SomMot'].mean())
        vis.append(df_group['Vis'].mean())

    age_groups = ['<60', '60-70', '70-80', '>80']
    x = np.arange(len(age_groups))  # Numeric positions for the x-axis
    # Labels and colors for each category
    categories = ['Default', 'Cont', 'Limbic', 'SalVenAttn', 'DorsAttn', 'SomMot', 'Vis']
    data = [default, cont, limbic, salvenattn, dorsattn, sommot, vis]
    colors = ['brown', 'orange', 'olive', 'magenta', 'green', 'teal', 'purple']

    fig, axes = plt.subplots(nrows=len(categories), ncols=1, figsize=(6, 6), sharex=True)

    for ax, y, label, color in zip(axes, data, categories, colors):
        ax.plot(x, y, label=label, color='k', marker='s', linestyle='-')
        if not svg:
            ax.set_ylabel(label, rotation=0, labelpad=30, color=color, fontsize=10)  # Network label
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(min(y) - 0.08, max(y) + 0.08)  # Adjust y-axis limits for better appearance

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(age_groups)

    plt.tight_layout(h_pad=-0.85)
    if svg:
        plt.xticks([])
        plt.savefig('figure1/1d/figures_1d_2.svg', format='svg')
    else:
        plt.savefig('figure1/1d/figures_1d.png', format='png')
    return fig, axes

def plot_bar_chart_1c(df, svg=False):
    network_name7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

    df_groups = [df_group for _, df_group in df.groupby('PTGENDER')]
    means = [df_groups[0][network].mean() - df_groups[1][network].mean() for network in network_name7]

    categories = ['SalVentAttn', 'Vis', 'DorsAttn', 'Cont', 'SomMot', 'Default', 'Limbic']
    colors = ['purple', 'green', 'teal', 'olive', 'orange', 'brown', 'magenta']

    sorted_means, sorted_categories = zip(*sorted(zip(means, categories)))

    means_df = pd.DataFrame({'Network': sorted_categories, 'Mean Difference': sorted_means})
    means_df.to_csv('figure1/1c/bar_plot.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(x=sorted_categories, y=sorted_means, color='red', ax=ax)

    if not svg:
        ax.set_xticklabels(categories, fontsize=12, color='black')
        ax.set_ylabel('Values', fontsize=12)
    else:
        ax.set_xticks([])  # Hides the labels
        ax.set_yticks([])  # Hides the labels
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    if svg:
        fig.savefig('figure1/1c/figures_1c2.svg', format='svg')
    else:
        fig.savefig('figure1/1c/figures_1c2.png', format='png')
    # plt.show()
    return fig, ax

def figure_1a_barplot():
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df_ = fig1.utils.preprocess_df(data_path)
    X, Y, Z = fig1.utils.get_input_output_confounder(df_)

    parcellation_df = pd.read_csv(os.path.join(os.getcwd(), "..", 'data/7network.csv'))
    network = 7
    Z.reset_index(drop=True, inplace=True)
    Y.reset_index(drop=True, inplace=True)

    Y_i = Y.iloc[:, :].values
    Y_scaler = StandardScaler()
    Y_i = Y_scaler.fit_transform(Y_i)

    coef_df = pd.DataFrame(columns=config.networks)
    model = LinearRegression()
    model.fit(X, Y_i)
    coef_ = model.coef_
    for i in range(network):
        col_network = list(parcellation_df[parcellation_df.network == i].index.values)
        col_network = [x + 1 for x in col_network]
        coef_df[config.networks[i]] = np.sum(coef_[:, col_network], axis=1)

    coef_df = (coef_df - coef_df.min()) / (coef_df.max() - coef_df.min())
    coef_df = coef_df *2 -1
    coef_df.to_csv("figure1/barplot_data.csv", index=False)
    return coef_df

if __name__ == '__main__':
    figure_1a_barplot()
    # plot_fig_1c()
    # plot_fig_1d()
    # plot_fig_1e()
    # plot_fig_1f()
    # figure1_test_statistic_difference_by_age_group()
