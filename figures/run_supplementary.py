import matplotlib.pyplot as plt
import os
import sys


sys.path.append(os.path.join(os.getcwd(), "..","dev"))

from cross_validator import CrossValidator

import numpy as np
import pandas as pd
from typing import Tuple
import figure6 as fig6
import figure3 as fig3
import figure1 as fig1
import figure4 as fig4
import simulation
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline,make_pipeline
from rePLS import rePLS,rePCR,reMLR
from scipy.stats import t,pearsonr
import random
from scipy import io


def compare_rePLS_vs_PLS():
    data_path = os.path.join(os.getcwd(), "..", 'data/ALL_3.csv')
    df = fig1.utils.preprocess_df(data_path)

    n_components = 5
    n_splits = 10
    n_repeats = 1
    random_state = 1
    out_dir = './supplementary/results'
    cv = CrossValidator(n_splits=n_splits, n_repeats=n_repeats,
                        stratified=True, random_state=random_state)

    methods = ["rePLS", "PLS"]
    for method in methods:
        mean_PQ, mean_P = fig4.utils.k_fold_prediction_PQ(df, cv, out_dir, n_components, random_state, n_splits, method)
        #save
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir,f'compare_rePLS_vs_PLS_method_{method}_2.mat')
        io.savemat(path, {'P': mean_P, 'PQ': mean_PQ})
        print(f"Saved {method}")
    return

if __name__ == '__main__':

    compare_rePLS_vs_PLS()
    print("Done!")


