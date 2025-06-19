import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import figures.figure5 as fig5
import figures.figure1 as fig1
from cross_validator import CrossValidator

def plot_fig_5b(show_plot=False):
    out_dir = "figure5/5b/"
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(csv_dir, "CN_group.csv")) and os.path.exists(os.path.join(csv_dir, "sMCI_group.csv")) and os.path.exists(os.path.join(csv_dir, "pMCI_group.csv")) and os.path.exists(os.path.join(csv_dir, "AD_group.csv")):
        CN_df = pd.read_csv(os.path.join(csv_dir, "CN_group.csv"))
        sMCI_df = pd.read_csv(os.path.join(csv_dir, "sMCI_group.csv"))
        pMCI_df = pd.read_csv(os.path.join(csv_dir, "pMCI_group.csv"))
        AD_df = pd.read_csv(os.path.join(csv_dir, "AD_group.csv"))

        melted_CN_df = CN_df.reset_index().melt(id_vars=["index"], var_name="Month", value_name="Diagnosis")
        melted_CN_df["group"] = "CN"
        melted_sMCI_df = sMCI_df.reset_index().melt(id_vars=["index"], var_name="Month", value_name="Diagnosis")
        melted_sMCI_df["group"] = "sMCI"
        melted_pMCI_df = pMCI_df.reset_index().melt(id_vars=["index"], var_name="Month", value_name="Diagnosis")
        melted_pMCI_df["group"] = "pMCI"
        melted_AD_df = AD_df.reset_index().melt(id_vars=["index"], var_name="Month", value_name="Diagnosis")
        melted_AD_df["group"] = "AD"

        combine_df = pd.concat([melted_CN_df, melted_sMCI_df, melted_pMCI_df, melted_AD_df])[
            ["Month", "Diagnosis", "group"]]
        combine_df = combine_df.reset_index()

        plt.figure(figsize=(18, 6))

        custom_palette = {"CN": "#b5c7e7", "sMCI": "#c5deb5", "pMCI": "#ffbf91", "AD": "#f18c8d"}

        # Create the boxplot
        sns.boxplot(
            data=combine_df,
            x="Month",
            y="Diagnosis",
            hue="group",
            palette=custom_palette,  # Custom colors
            dodge=True,  # Increase space between groups
            showfliers=False,  # Hide outliers
            width=0.6,
            boxprops={"edgecolor": "black", "linewidth": 0.5},  # Set box border color
            whiskerprops={"color": "black", "linewidth": 1},  # Set whisker color
            capprops={"color": "black", "linewidth": 1},  # Set cap color
            medianprops={"color": "black", "linewidth": 1}  # Set median line color

        )

        sns.despine(top=True, right=True)

        plt.xlabel("")
        plt.ylabel("")
        plt.title("")
        plt.gca().set_xticklabels(["" for _ in plt.gca().get_xticks()])
        plt.gca().set_yticklabels(["" for _ in plt.gca().get_yticks()])
        # # Save the figure as an SVG file
        svg_filename = "fig5_b.svg"
        plt.savefig(os.path.join(out_dir,svg_filename) , format="svg", bbox_inches="tight")
        print(f"Figure saved to {os.path.join(out_dir, svg_filename)}")
        if show_plot:
            plt.show()

def plot_fig_5c(show_plot=False):
    out_dir = "figure5/5c/"
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(csv_dir, "CN_group.csv")) and os.path.exists(os.path.join(csv_dir, "sMCI_group.csv")) and os.path.exists(os.path.join(csv_dir, "pMCI_group.csv")) and os.path.exists(os.path.join(csv_dir, "AD_group.csv")):
        CN_df = pd.read_csv(os.path.join(csv_dir, "CN_group.csv"))
        sMCI_df = pd.read_csv(os.path.join(csv_dir, "sMCI_group.csv"))
        pMCI_df = pd.read_csv(os.path.join(csv_dir, "pMCI_group.csv"))
        AD_df = pd.read_csv(os.path.join(csv_dir, "AD_group.csv"))

        # Melt dataframe (index as id_vars)
        melted_CN_df = CN_df.reset_index().melt(id_vars=["index"], var_name="Month", value_name="Diagnosis")
        melted_CN_df["group"] = "CN"
        melted_sMCI_df = sMCI_df.reset_index().melt(id_vars=["index"], var_name="Month", value_name="Diagnosis")
        melted_sMCI_df["group"] = "sMCI"
        melted_pMCI_df = pMCI_df.reset_index().melt(id_vars=["index"], var_name="Month", value_name="Diagnosis")
        melted_pMCI_df["group"] = "pMCI"
        melted_AD_df = AD_df.reset_index().melt(id_vars=["index"], var_name="Month", value_name="Diagnosis")
        melted_AD_df["group"] = "AD"
        combine_df = pd.concat([melted_CN_df, melted_sMCI_df, melted_pMCI_df, melted_AD_df])[
            ["Month", "Diagnosis", "group"]]
        combine_df = combine_df.reset_index()

        custom_palette = {"CN": "#b5c7e7", "sMCI": "#c5deb5", "pMCI": "#ffbf91", "AD": "#f18c8d"}

        lineplot_data = combine_df.groupby(["Month", "group"])["Diagnosis"].agg(["mean", "std", "count"]).reset_index()
        lineplot_data["se"] = lineplot_data["std"] / np.sqrt(lineplot_data["count"])


        month_order = ["bl", "m06", "m12", "m18", "m24", "m36", "m48", "m60", "m72", "m84", "m96"]
        lineplot_data["Month"] = pd.Categorical(lineplot_data["Month"], categories=month_order, ordered=True)
        lineplot_data = lineplot_data.sort_values("Month")  # Ensure correct order

        month_numeric = {month: idx for idx, month in enumerate(month_order)}
        lineplot_data["Month_numeric"] = lineplot_data["Month"].map(month_numeric)

        plt.figure(figsize=(18, 6))
        sns.lineplot(
            data=lineplot_data,
            x="Month_numeric",
            y="mean",
            hue="group",
            palette=custom_palette,
            marker="o",  # Markers for each time point
            linewidth=4,
            # ci=None  # Remove automatic seaborn CI to manually add shaded regions
        )

        custom_palette = {"CN": "#4472C4", "sMCI": "#71AD47", "pMCI": "#ED7D31", "AD": "#FF0000"}
        # Add confidence intervals manually
        for group in lineplot_data["group"].unique():
            subset = lineplot_data[lineplot_data["group"] == group]
            plt.fill_between(subset["Month_numeric"], subset["mean"] - subset["se"] * 1.96,
                             subset["mean"] + subset["se"] * 1.96,
                             color=custom_palette[group], alpha=0.2)  # Alpha for transparency

        # Replace x-axis numeric ticks with original categorical month labels
        plt.xticks(ticks=list(month_numeric.values()), labels=month_order, rotation=45)

        # Remove spines on the top and right
        sns.despine(top=True, right=True)

        # # Format the plot

        # # Customize legend

        plt.xlabel("")
        plt.ylabel("")
        plt.title("")
        plt.yticks([0.5, 1, 1.5, 2])
        plt.gca().set_xticklabels(["" for _ in plt.gca().get_xticks()])
        plt.gca().set_yticklabels(["" for _ in plt.gca().get_yticks()])
        # # Save the figure as an SVG file
        svg_filename = "fig5_c.svg"
        plt.savefig(os.path.join(out_dir, svg_filename), format="svg", bbox_inches="tight")
        print(f"Figure saved to {os.path.join(out_dir, svg_filename)}")
        if show_plot:
            plt.show()

def prepare_data_fig_5bc():
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)
    n_components = 3
    n_splits = 10
    n_repeats = 50
    random_state = 1
    out_dir = './figure5/5b'
    csv_dir = os.path.join(out_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    if os.path.exists(os.path.join(csv_dir, "df_result_shift_repeated.csv")) and os.path.exists(os.path.join(csv_dir, "df_result_shift_label_repeated.csv")):
        df_result_shift_repeated = pd.read_csv(os.path.join(csv_dir, "df_result_shift_repeated.csv"))
        df_result_shift_label_repeated = pd.read_csv(os.path.join(csv_dir, "df_result_shift_label_repeated.csv"))
    else:
        cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                            stratified=True, random_state=random_state)
        df_result_shift_repeated, df_result_shift_label_repeated = fig5.utils.repeated_k_fold_longitudinal_disease_prediction(df, cv, out_dir, n_components, random_state, n_splits)
        #write to csv
        path = os.path.join(out_dir, 'df_result_shift_repeated.csv')
        df_result_shift_repeated.to_csv(path,index=False)
        print(f"Save to {path}")

        path = os.path.join(out_dir, 'df_result_shift_label_repeated.csv')
        df_result_shift_label_repeated.to_csv(path,index=False)
        print(f"Save to {path}")

    df = df_result_shift_label_repeated
    data_df = df_result_shift_repeated
    cn_df = data_df[(df.iloc[:, 0] == 0) & (df.iloc[:, 1:].fillna(0).eq(0).all(axis=1))]
    cn_df.to_csv(os.path.join(csv_dir, 'CN_group.csv'), index=False)
    smci_df = data_df[(df.iloc[:, 0] == 1) & (df.iloc[:, 1:].fillna(1).eq(1).all(axis=1))]
    smci_df.to_csv(os.path.join(csv_dir, 'sMCI_group.csv'), index=False)
    pmci_df = data_df[(df.iloc[:, 0] == 1) & (df.iloc[:, 1:].fillna(0).eq(2).any(axis=1))]
    pmci_df.to_csv(os.path.join(csv_dir, 'pMCI_group.csv'), index=False)
    ad_df = data_df[df.iloc[:, 0] == 2]
    ad_df.to_csv(os.path.join(csv_dir, 'AD_group.csv'), index=False)
    return df_result_shift_repeated, df_result_shift_label_repeated

def plot_fig_5de(show_plot=False):
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)
    n_components = 3
    n_splits = 10
    n_repeats = 20
    random_state = 1
    out_dir = './figure5/5d'
    csv_path = os.path.join(out_dir, '..')
    os.makedirs(csv_path, exist_ok=True)
    if not os.path.exists(os.path.join(csv_path, "P_longitudinal.csv")) and not os.path.exists(os.path.join(csv_path, "Q_longitudinal.csv")) and not os.path.exists(os.path.join(csv_path, "alpha_longitudinal.csv")):
        cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                            stratified=True, random_state=random_state)
        mean_P, mean_alpha,mean_Q = fig5.utils.k_fold_disease_prediction(df, cv, out_dir, n_components, random_state, n_splits)
        pd.DataFrame(mean_P).to_csv(os.path.join(csv_path, "P_longitudinal.csv"), index=False)
        pd.DataFrame(mean_Q).to_csv(os.path.join(csv_path, "Q_longitudinal.csv"), index=False)
        pd.DataFrame(mean_alpha).to_csv(os.path.join(csv_path, "alpha_longitudinal.csv"), index=False)
    else:
        mean_P = pd.read_csv(os.path.join(csv_path, "P_longitudinal.csv")).values
        # mean_Q = pd.read_csv(os.path.join(csv_path, "Q_longitudinal.csv")).values
        # mean_alpha = pd.read_csv(os.path.join(csv_path, "alpha_longitudinal.csv")).values



if __name__ == '__main__':
    # prepare_data_fig_5bc()
    # plot_fig_5b()
    # plot_fig_5c()
    plot_fig_5de()






