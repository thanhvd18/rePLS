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

import simulation
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.pipeline import Pipeline,make_pipeline
# from rePLS import rePLS,rePCR,reMLR
# from scipy.stats import t,pearsonr
# import random



def run_s2_a():
    N = 100
    I = 20
    J = 2

    SNR_YZ = -10
    Z_affect_regions = 0.05
    N_repeats = 5
    methods = ["PLS", "rePLS"]
    result_df = pd.DataFrame(columns=["method", "SNR_XZ","SNR_XY", "r", "MSE", "p_value"])
    SNR_XZs = [-10, 0, 10]
    SNR_XY_range = np.linspace(-50, 50, 20, endpoint=True)
    for method in methods:
        for SNR_XZ in SNR_XZs:
            for i_xy, SNR_XY in enumerate(SNR_XY_range):
                X, X_true, Y, Y_true, Z = simulation.utils.generate_data(N, I, J, SNR_XZ, Z_affect_regions, SNR_YZ, SNR_XY,
                                                                         random_seed=1)
                df_repeated = simulation.utils.boostrapping_prediction(X, Y, Y_true, Z, method, N_repeats)
                df_repeated['SNR_XY'] = SNR_XY
                df_repeated['SNR_XZ'] = SNR_XZ


                df_repeated['method'] = method
                df_repeated = pd.DataFrame([df_repeated])
                result_df = pd.concat([result_df, df_repeated], ignore_index=True)

    print(result_df)
    legends = []
    for (method, snr_xz), group in result_df.groupby(['method', 'SNR_XZ']):
        legends.append(f'{method}, SNR_XZ={snr_xz}')
        plt.plot(group['SNR_XY'], group['r'], marker='o', label=f'{method}, SNR_XZ={snr_xz}')
    plt.legend(legends)
        # plt.title(method)
    plt.show()
    print("Done!")

def run_s2_b():
    N = 100
    I = 20
    J = 2


    Z_affect_regions = 0.05
    N_repeats = 5
    methods = ["PLS", "rePLS"]
    result_df = pd.DataFrame(columns=["method", "SNR_YZ","SNR_XZ","SNR_XY", "r", "MSE", "p_value"])
    SNR_XZs = [-10, 0, 10]
    SNR_YZ_range = [0, 10]
    SNR_XY_range = np.linspace(-50, 50, 20, endpoint=True)
    for method in methods:
        for SNR_YZ in SNR_YZ_range:
            for i_xy, SNR_XY in enumerate(SNR_XY_range):
                for SNR_XZ in SNR_XZs:
                    print(f"method={method}, SNR_YZ={SNR_YZ}, SNR_XY={SNR_XY}, SNR_XZ={SNR_XZ}")
                    X, X_true, Y, Y_true, Z = simulation.utils.generate_data(N, I, J, SNR_XZ, Z_affect_regions, SNR_YZ, SNR_XY,
                                                                             random_seed=1)
                    df_repeated = simulation.utils.boostrapping_prediction(X, Y, Y_true, Z, method, N_repeats)
                    df_repeated['SNR_XY'] = SNR_XY
                    df_repeated['SNR_XZ'] = SNR_XZ
                    df_repeated['SNR_YZ'] = SNR_YZ


                    df_repeated['method'] = method
                    df_repeated = pd.DataFrame([df_repeated])
                    result_df = pd.concat([result_df, df_repeated], ignore_index=True)

    print(result_df)
    legends = []
    for (method, snr_xz,snr_yz), group in result_df.groupby(['method', 'SNR_XZ','SNR_YZ']):
        legends.append(f'{method}, SNR_XZ={snr_xz}')
        plt.plot(group['SNR_XY'], group['r'], marker='o')
    plt.legend(legends)
        # plt.title(method)
    plt.show()
    print("Done!")

def run_s2_c():
    N = 100
    I = 20
    J = 2


    Z_affect_regions = 0.05
    N_repeats = 5
    methods = ["PLS", "rePLS"]
    result_df = pd.DataFrame(columns=["method", "SNR_YZ","SNR_XZ","SNR_XY", "r", "MSE", "p_value"])
    SNR_XZs = [0, 10]
    SNR_YZ_range = [10]
    SNR_XY_range = np.linspace(-50, 50, 20, endpoint=True)
    for method in methods:
        for SNR_YZ in SNR_YZ_range:
            for i_xy, SNR_XY in enumerate(SNR_XY_range):
                for SNR_XZ in SNR_XZs:
                    print(f"method={method}, SNR_YZ={SNR_YZ}, SNR_XY={SNR_XY}, SNR_XZ={SNR_XZ}")
                    X, X_true, Y, Y_true, Z = simulation.utils.generate_data(N, I, J, SNR_XZ, Z_affect_regions, SNR_YZ, SNR_XY,
                                                                             random_seed=1)
                    df_repeated = simulation.utils.boostrapping_prediction(X, Y, Y_true, Z, method, N_repeats)
                    df_repeated['SNR_XY'] = SNR_XY
                    df_repeated['SNR_XZ'] = SNR_XZ
                    df_repeated['SNR_YZ'] = SNR_YZ


                    df_repeated['method'] = method
                    df_repeated = pd.DataFrame([df_repeated])
                    result_df = pd.concat([result_df, df_repeated], ignore_index=True)

    print(result_df)
    legends = []
    for (method, snr_xz,snr_yz), group in result_df.groupby(['method', 'SNR_XZ','SNR_YZ']):
        legends.append(f'{method}, SNR_XZ={snr_xz}')
        plt.plot(group['SNR_XY'], group['r'], marker='o')
    plt.legend(legends)
        # plt.title(method)
    plt.show()
    print("Done!")
def run_s3():
    N = 100
    I = 20
    J = 8


    Z_affect_regions = 0.2
    N_repeats = 5
    methods = ["PLS", "rePLS"]
    result_df = pd.DataFrame(columns=["method", "SNR_YZ","SNR_XZ","SNR_XY", "r", "MSE", "p_value"])
    SNR_XZs = [-10,0,10]
    SNR_YZ_range = np.linspace(-50, 50, 20, endpoint=True)
    SNR_XY_range = [-10,0,10]
    for method in methods:
        for SNR_YZ in SNR_YZ_range:
            for i_xy, SNR_XY in enumerate(SNR_XY_range):
                for SNR_XZ in SNR_XZs:
                    print(f"method={method}, SNR_YZ={SNR_YZ}, SNR_XY={SNR_XY}, SNR_XZ={SNR_XZ}")
                    X, X_true, Y, Y_true, Z = simulation.utils.generate_data(N, I, J, SNR_XZ, Z_affect_regions, SNR_YZ, SNR_XY,
                                                                             random_seed=1)
                    df_repeated = simulation.utils.boostrapping_prediction(X, Y, Y_true, Z, method, N_repeats)
                    df_repeated['SNR_XY'] = SNR_XY
                    df_repeated['SNR_XZ'] = SNR_XZ
                    df_repeated['SNR_YZ'] = SNR_YZ


                    df_repeated['method'] = method
                    df_repeated = pd.DataFrame([df_repeated])
                    result_df = pd.concat([result_df, df_repeated], ignore_index=True)

    print(result_df)
    legends = []
    for (method, snr_xy,snr_xz), group in result_df.groupby(['method', 'SNR_XY','SNR_XZ']):
        legends.append(f'{method}, SNR_XY={snr_xy}, SNR_YZ={snr_xz}')
        plt.plot(group['SNR_YZ'], group['r'], marker='o')
    plt.legend(legends)
        # plt.title(method)
    plt.show()
    print("Done!")

def run_s4():
    N = 100
    I = 20
    J = 8


    Z_affect_regions = 0.2
    N_repeats = 5
    methods = ["PLS", "rePLS"]
    result_df = pd.DataFrame(columns=["method", "SNR_YZ","SNR_XZ","SNR_XY", "r", "MSE", "p_value"])
    SNR_XZs = np.linspace(-50, 50, 20, endpoint=True)
    SNR_YZ_range = [-10, 0, 10]
    SNR_XY_range = [-10,0,10]
    for method in methods:
        for SNR_YZ in SNR_YZ_range:
            for i_xy, SNR_XY in enumerate(SNR_XY_range):
                for SNR_XZ in SNR_XZs:
                    print(f"method={method}, SNR_YZ={SNR_YZ}, SNR_XY={SNR_XY}, SNR_XZ={SNR_XZ}")
                    X, X_true, Y, Y_true, Z = simulation.utils.generate_data(N, I, J, SNR_XZ, Z_affect_regions, SNR_YZ, SNR_XY,
                                                                             random_seed=1)
                    df_repeated = simulation.utils.boostrapping_prediction(X, Y, Y_true, Z, method, N_repeats)
                    df_repeated['SNR_XY'] = SNR_XY
                    df_repeated['SNR_XZ'] = SNR_XZ
                    df_repeated['SNR_YZ'] = SNR_YZ


                    df_repeated['method'] = method
                    df_repeated = pd.DataFrame([df_repeated])
                    result_df = pd.concat([result_df, df_repeated], ignore_index=True)

    print(result_df)
    legends = []
    for (method, snr_xy,snr_yz), group in result_df.groupby(['method', 'SNR_XY','SNR_YZ']):
        legends.append(f'{method}, SNR_XY={snr_xy}, SNR_YZ={snr_yz}')
        plt.plot(group['SNR_XZ'], group['r'], marker='o')
    plt.legend(legends)
        # plt.title(method)
    plt.show()
    print("Done!")

if __name__ == '__main__':
    # run_s2_a()
    # run_s2_b()
    # run_s2_c()
    # run_s3()
    # run_s4()
    print("Done!")


