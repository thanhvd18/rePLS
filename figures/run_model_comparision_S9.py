import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..', 'dev'))

from cross_validator import CrossValidator

import numpy as np
import pandas as pd
from typing import Tuple
import figure3 as fig3
import figure1 as fig1
import seaborn as sns

def plot_fig_S9():
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    n_components = 5
    n_splits = 10
    n_repeats = 1
    random_state = 1
    out_dir = './figure3/results'

    result_df = pd.DataFrame(columns= ["method", 'outcome', 'r', 'MSE', 'p_value'])
    for i,method in enumerate(["rePLS", "PLS", "PCR", "rePCR", "LR", "reMLR"]):

        if method == "rePLS" or method == "PLS":
            n_components = 5 #select by cross-validation
        elif method == "rePCR" or method == "PCR":
            n_components = 20 #select by cross-validation
        cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                            stratified=False, random_state=random_state)

        stat_df, predict_result_df, combine_stat_df = fig3.utils.k_fold_prediction(df, cv, out_dir, method, n_components=n_components,
                                                                                   random_state=random_state, n_splits=n_splits)
        combine_stat_df['method'] = method
        if i == 0:
            result_df = combine_stat_df
        else:
            result_df = pd.concat([result_df, combine_stat_df])

    sns.catplot(
        data=result_df, kind="bar",
        x="outcome", y="r", hue="method")
    plt.show()
    return

if __name__ == '__main__':
    # combine_stat_df = plot_fig_3a()
    # print(combine_stat_df)
    plot_fig_S9()



