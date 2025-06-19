import os
import sys


import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import pearsonr
import scipy
import seaborn as sns
import matplotlib.pyplot as plt


import figures.figure6 as fig6
import figures.figure3 as fig3
import figures.figure1 as fig1

from cross_validator import CrossValidator


def plot_fig_6b(show_plot=False):
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')

    df = fig1.utils.preprocess_df(data_path)

    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    n_components = 5
    n_splits = 10
    n_repeats = 1
    random_state = 1
    out_dir = './figure6/6b'
    csv_dir = os.path.join(out_dir, 'csv')
    os.makedirs(out_dir, exist_ok=True)
    dataset = "ADNI"
    method = 'rePLS'  # ["rePLS", "PLS", "PCR", "rePCR", "LR", "reMLR"]
    cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                        stratified=False, random_state=random_state)
    if not os.path.exists(os.path.join(csv_dir,f"predict_{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}")) and \
            not os.path.exists(os.path.join(csv_dir,f"combine_stat_{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}")) and \
            not os.path.exists(os.path.join(csv_dir,f"stat_t{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}")):
        stat_df, predict_result_df, combine_stat_df = fig3.utils.k_fold_prediction(df, cv, csv_dir, method, n_components,  random_state, n_splits, dataset=dataset)
        stat_df.to_csv(os.path.join(csv_dir,f"stat_t{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}.csv"), index=False)
        predict_result_df.to_csv(os.path.join(csv_dir,f"predict_{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}.csv"), index=False)
        combine_stat_df.to_csv(os.path.join(csv_dir,f"combine_stat_{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}.csv"), index=False)
    else:
        stat_df = pd.read_csv(os.path.join(csv_dir,f"stat_t{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}.csv"))
        predict_result_df = pd.read_csv(os.path.join(csv_dir,f"predict_{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}.csv"))
        combine_stat_df = pd.read_csv(os.path.join(csv_dir,f"combine_stat_{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}.csv"))

    plt.figure(figsize=(6, 6))
    sns.regplot(
        data=predict_result_df,
        x="outcome0",
        y="outcome0_pred",
        scatter=True,
        # color="blue"
    )

    r, p_value = pearsonr(predict_result_df["outcome0"], predict_result_df["outcome0_pred"])
    print(f"Correlation coefficient (r): {r}")
    print(f"P-value: {p_value}")

    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Regression Line (r={r:.2f}, p={p_value:.2e})")
    plt.savefig(os.path.join(out_dir, "CDRSB.png"))
    if show_plot:
        plt.show()

    plt.figure(figsize=(6, 6))
    sns.regplot(
        data=predict_result_df,
        x="outcome4",
        y="outcome4_pred",
        # hue="DX_encode",
        # palette="coolwarm",
        scatter=True,
        # color="blue"
    )

    r, p_value = pearsonr(predict_result_df["outcome4"], predict_result_df["outcome4_pred"])
    print(f"Correlation coefficient (r): {r}")
    print(f"P-value: {p_value}")

    # Set plot labels and title
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Regression Line (r={r:.2f}, p={p_value:.2e})")
    plt.savefig(os.path.join(out_dir, "MMSE.png"))
    if show_plot:
        plt.show()
    return

def plot_fig_6c(show_plot=False):
    data_path = os.path.join(os.getcwd(), "..", 'data/oasis_cross-sectional_thickness_combined.csv')
    df = fig6.utils.load_OASIS(data_path)
    X, Y, Z = fig6.utils.get_input_output_confounder_OASIS(df)
    n_components = 5
    n_splits = 10
    n_repeats = 1
    random_state = 1
    dataset = "OASIS"
    out_dir = './figure6/6c'
    csv_dir = os.path.join(out_dir, 'csv')
    os.makedirs(out_dir, exist_ok=True)
    method = 'rePLS'  # ["rePLS", "PLS", "PCR", "rePCR", "LR", "reMLR"]
    cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                        stratified=False, random_state=random_state)

    stat_df, predict_result_df, combine_stat_df = fig6.utils.k_fold_prediction_OASIS(df, cv, out_dir, method, n_components,
                                                            random_state, n_splits, dataset="OASIS")
    plt.figure(figsize=(6, 6))
    sns.regplot(
        data=predict_result_df,
        x="outcome0",
        y="outcome0_pred",
        scatter=True,
        # color="blue"
    )

    # Calculate correlation coefficient and p-value
    r, p_value = pearsonr(predict_result_df["outcome0"], predict_result_df["outcome0_pred"])
    print(f"Correlation coefficient (r): {r}")
    print(f"P-value: {p_value}")

    # Set plot labels and title
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Regression Line (r={r:.2f}, p={p_value:.2e})")
    plt.savefig(os.path.join(out_dir, "CDRSB.png"))
    if show_plot:
        plt.show()

    plt.figure(figsize=(6, 6))
    sns.regplot(
        data=predict_result_df,
        x="outcome1",
        y="outcome1_pred",
        # hue="DX_encode",
        # palette="coolwarm",
        scatter=True,
        # color="blue"
    )

    r, p_value = pearsonr(predict_result_df["outcome1"], predict_result_df["outcome1_pred"])
    print(f"Correlation coefficient (r): {r}")
    print(f"P-value: {p_value}")

    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Regression Line (r={r:.2f}, p={p_value:.2e})")
    plt.savefig(os.path.join(out_dir, "MMSE.png"))
    if show_plot:
        plt.show()
    return combine_stat_df

def plot_fig_6d(show_plot=False):
    data_path = os.path.join(os.getcwd(), "..", 'data/oasis_cross-sectional_thickness_combined.csv')
    df = fig6.utils.load_OASIS(data_path)
    X, Y, Z = fig6.utils.get_input_output_confounder_OASIS(df)
    n_components = 2
    n_splits = 10
    n_repeats = 1
    random_state = 1
    out_dir = './figure6/6d'
    os.makedirs(out_dir, exist_ok=True)
    method = 'rePLS'  # ["rePLS", "PLS", "PCR", "rePCR", "LR", "reMLR"]
    cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                        stratified=False, random_state=random_state)

    models, scaler_Xs, scaler_Ys = fig6.utils.k_fold_train_OASIS(df, cv, out_dir, method, n_components, random_state,
                                                                n_splits)
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)
    X, Y, Z = fig1.utils.get_input_output_confounder(df)

    predict_result_df = fig6.utils.test_ADNI(models, scaler_Xs, scaler_Ys, X, Y, Z, out_dir, method=method)
    # Y_df['CDR'] = Y[:, 0]
    # Y_df['MMSE'] = Y[:, 4]
    # Y_df['pred_CDR'] = Y_pred_mean[:, 0]
    # Y_df['pred_MMSE'] = Y_pred_mean[:, 1]
    plt.figure(figsize=(6, 6))
    sns.regplot(
        data=predict_result_df,
        x="CDR",
        y="pred_CDR",
        scatter=True,
        # color="blue"
    )

    r, p_value = pearsonr(predict_result_df["CDR"], predict_result_df["pred_CDR"])
    print(f"Correlation coefficient (r): {r}")
    print(f"P-value: {p_value}")

    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Regression Line (r={r:.2f}, p={p_value:.2e})")
    plt.savefig(os.path.join(out_dir, "CDRSB.png"))
    if show_plot:
        plt.show()

    plt.figure(figsize=(6, 6))
    sns.regplot(
        data=predict_result_df,
        x="MMSE",
        y="pred_MMSE",
        # hue="DX_encode",
        # palette="coolwarm",
        scatter=True,
        # color="blue"
    )

    r, p_value = pearsonr(predict_result_df["MMSE"], predict_result_df["pred_MMSE"])
    print(f"Correlation coefficient (r): {r}")
    print(f"P-value: {p_value}")

    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Regression Line (r={r:.2f}, p={p_value:.2e})")
    plt.savefig(os.path.join(out_dir, "MMSE.png"))
    if show_plot:
        plt.show()

def plot_fig_6e(show_plot=False):
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')

    df = fig1.utils.preprocess_df(data_path)

    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    n_components = 5
    n_splits = 10
    n_repeats = 1
    random_state = 1
    out_dir = './figure6/6e'
    os.makedirs(out_dir, exist_ok=True)
    method = 'rePLS'  # ["rePLS", "PLS", "PCR", "rePCR", "LR", "reMLR"]
    cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                        stratified=False, random_state=random_state)

    models, scaler_Xs, scaler_Ys = fig6.utils.k_fold_train_ADNI(df, cv, out_dir, method, n_components, random_state, n_splits)

    data_path = os.path.join(os.getcwd(), "..", 'data/oasis_cross-sectional_thickness_combined.csv')
    df = fig6.utils.load_OASIS(data_path)
    X, Y, Z = fig6.utils.get_input_output_confounder_OASIS(df)
    predict_result_df = fig6.utils.test_OASIS(models, scaler_Xs, scaler_Ys, X, Y, Z,out_dir, components=[0],method=method)

    plt.figure(figsize=(6, 6))
    sns.regplot(
        data=predict_result_df,
        x="CDR",
        y="pred_CDR",
        scatter=True,
        # color="blue"
    )

    r, p_value = pearsonr(predict_result_df["CDR"], predict_result_df["pred_CDR"])
    print(f"Correlation coefficient (r): {r}")
    print(f"P-value: {p_value}")

    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Regression Line (r={r:.2f}, p={p_value:.2e})")
    plt.savefig(os.path.join(out_dir, "CDRSB.png"))
    if show_plot:
        plt.show()

    plt.figure(figsize=(6, 6))
    sns.regplot(
        data=predict_result_df,
        x="MMSE",
        y="pred_MMSE",
        # hue="DX_encode",
        # palette="coolwarm",
        scatter=True,
        # color="blue"
    )

    r, p_value = pearsonr(predict_result_df["MMSE"], predict_result_df["pred_MMSE"])
    print(f"Correlation coefficient (r): {r}")

    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Regression Line (r={r:.2f}, p={p_value:.2e})")
    plt.savefig(os.path.join(out_dir, "MMSE.png"))
    if show_plot:
        plt.show()


def plot_fig_6f(): # common/unique
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')

    df = fig1.utils.preprocess_df(data_path)

    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    n_components = 5
    n_splits = 10
    n_repeats = 50
    random_state = 1
    thresh_per = 0.25
    intersection_perc = 0.9
    out_dir = './figure6/6f'
    csv_dir = os.path.join(out_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    method = 'rePLS'  # ["rePLS", "PLS", "PCR", "rePCR", "LR", "reMLR"]
    cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                        stratified=False, random_state=random_state)

    ADNI_PQ_path, ADNI_P_path = fig6.utils.consistent_brainmap_k_fold_repeated_ADNI(df, cv, out_dir, method, n_components, random_state=random_state, n_splits=n_splits,  thresh_per=thresh_per, intersection_perc=intersection_perc)

    data_path = os.path.join(os.getcwd(), "..", 'data/oasis_cross-sectional_thickness_combined.csv')
    df = fig6.utils.load_OASIS(data_path)
    X, Y, Z = fig6.utils.get_input_output_confounder_OASIS(df)
    n_components = 2
    method = 'rePLS'  # ["rePLS", "PLS", "PCR", "rePCR", "LR", "reMLR"]
    cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                        stratified=False, random_state=random_state)

    OASIS_PQ_path, OASIS_P_path = fig6.utils.consistent_brainmap_k_fold_repeated_OASIS(df, cv, out_dir, method, n_components, random_state=random_state, n_splits=n_splits,  thresh_per=thresh_per, intersection_perc=intersection_perc)
    CDR_weight, MMSE_weight = fig6.utils.common_unique_regions(OASIS_P_path,ADNI_P_path, out_dir)

    CDR_file_path = os.path.join(csv_dir, "CDR_weight.csv")
    pd.DataFrame(CDR_weight).to_csv(CDR_file_path, index=False, header=False)
    MMSE_file_path = os.path.join(csv_dir, "MMSE_weight.csv")
    pd.DataFrame(MMSE_weight).to_csv(MMSE_file_path, index=False, header=False)



if __name__ == '__main__':
    plot_fig_6b()
    # plot_fig_6c()
    # plot_fig_6d()
    # plot_fig_6e()
    # plot_fig_6f()


