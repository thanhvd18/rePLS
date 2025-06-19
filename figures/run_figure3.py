import os
import sys

import seaborn as sns
from icecream import ic
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt


import config
import figures.figure3 as fig3
import figures.figure1 as fig1
from cross_validator import CrossValidator


def plot_fig_3a(out_dir='./figure3/3a',random_state=1):
    out_dir = f"{out_dir}_{random_state}"
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
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
        plt.show()
    return combine_stat_df

def plot_fig_3b(out_dir='./figure3/3b',random_state=1):
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
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
    plt.show()
    return

def plot_fig_3b_supplementary():
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
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
    plot_fig_3b()
    # plot_fig_3b_supplementary()



