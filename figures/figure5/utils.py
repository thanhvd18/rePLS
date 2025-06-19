import os
import sys

from sklearn.cluster import mean_shift

sys.path.append((os.path.join(os.getcwd(), 'figure1')))
import figure1 as fig1
import figure3 as fig3
from figure3.utils import cal_correlation_MSE_regression

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


from rePLS import rePLS, rePCR, reMLR
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline,make_pipeline

from typing import Tuple, List, Dict, Any, Optional
sys.path.append(os.path.join(os.getcwd(), '..', 'dev'))

from cross_validator import CrossValidator
from icecream import ic
from sklearn.utils import resample
from tqdm import tqdm

def k_fold_disease_prediction(df: pd.DataFrame,cv: CrossValidator,out_dir: str,n_components: int,random_state: int,n_splits: int) -> Tuple[np.ndarray, np.ndarray]:
    df, selected_subjects, labels = fig1.utils.categorize_disease_group(df)
    _, _, Z = fig1.utils.get_input_output_confounder(df)
    selected_subjects = df['SubjectID'].unique()
    confounders = Z.columns.values

    Ps = []
    alphas = []
    Qs = []
    for fold, (train_index, test_index) in enumerate(cv.get_splits(selected_subjects,labels)):
        print(f"Processing fold {fold}")
        train_subjects = np.array(selected_subjects)[train_index]
        test_subjects = np.array(selected_subjects)[test_index]

        df_train = df[df['SubjectID'].isin(train_subjects)]
        df_test = df[df['SubjectID'].isin(test_subjects)]

        X_train = np.vstack(df_train['Schaefer_200_7'].apply(eval))
        # Y_train = np.array(df_train['stage'])
        Y_train = np.vstack(df_train['DX_one_hot'])
        Z_train = np.array(df_train[confounders], dtype=float)

        X_test = np.vstack(df_test['Schaefer_200_7'].apply(eval))
        Y_test = np.array(df_test['stage'])
        # Y_test_ = np.copy(Y_test)
        Y_test = np.vstack(df_test['DX_one_hot'])
        Z_test = np.array(df_test[confounders], dtype=float)

        X_train, X_test = X_train[:, :], X_test[:, :]

        # normalize X, Y scaler
        X_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        # Y_scaler = StandardScaler()
        # Y_train = Y_scaler.fit_transform(Y_train)
        # Y_test = Y_scaler.transform(Y_test)
        Y_train = np.vstack(Y_train)
        Y_test = np.vstack(Y_test)



        model = rePLS(n_components=n_components, Z=Z_train)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test, Z=Z_test)

        P = model.P
        Q = model.Q
        alpha = np.linalg.pinv(P) @ model.residual_model.coef_.T @ np.linalg.pinv(Q.T)

        Ps.append(P)
        Qs.append(Q)
        alphas.append(alpha)
    mean_P = np.mean(Ps, axis=0)
    mean_alpha = np.mean(alphas, axis=0)
    mean_Q = np.mean(Qs, axis=0)
    return mean_P, mean_alpha, mean_Q


def repeated_k_fold_longitudinal_disease_prediction(df: pd.DataFrame, cv: CrossValidator, out_dir: str, n_components: int, random_state: int,
                              n_splits: int) -> Tuple[np.ndarray, np.ndarray]:
    df, selected_subjects, labels = fig1.utils.categorize_disease_group(df)
    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    selected_subjects = df['SubjectID'].unique()
    outcomes = Y.columns.values
    confounders = Z.columns.values
    n_outcomes = Y.shape[1]
    df['stage'] = df['DX'].map({'CN': 0, 'MCI': 1, 'AD': 2})

    Y = df['stage'].apply(lambda x: np.array((x)), 0)

    timepoints = ['bl', 'm06', 'm12', 'm18', 'm24', 'm36', 'm48', 'm60', 'm72',
                  'm84', 'm96', 'm108']
    n_timepoint = len(timepoints)
    df_result_shift_repreated = pd.DataFrame()
    df_result_shift_label_repreated = pd.DataFrame()
    for fold, (train_index, test_index) in enumerate(cv.get_splits(selected_subjects, labels)):
        print(f"Processing fold {fold}")
        train_subjects = np.array(selected_subjects)[train_index]
        test_subjects = np.array(selected_subjects)[test_index]

        df_train = df[df['SubjectID'].isin(train_subjects)]
        df_test = df[df['SubjectID'].isin(test_subjects)]

        X_train = np.vstack(df_train['Schaefer_200_7'].apply(eval))
        Y_train = np.array(df_train['stage'])
        Z_train = np.array(df_train[confounders], dtype=float)

        X_test = np.vstack(df_test['Schaefer_200_7'].apply(eval))
        Y_test = np.array(df_test['stage'])
        Y_test_ = np.copy(Y_test)
        Z_test = np.array(df_test[confounders], dtype=float)

        X_train, X_test = X_train[:, :], X_test[:, :]

        # normalize X, Y scaler
        X_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        # Y_scaler = StandardScaler()
        # Y_train = Y_scaler.fit_transform(Y_train)
        # Y_test = Y_scaler.transform(Y_test)
        Y_train = np.vstack(Y_train)
        Y_test = np.vstack(Y_test)
        model = rePLS(n_components=n_components, Z=Z_train)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test, Z=Z_test)

        df_test['stage_pred'] = y_pred
        subjects_test = df_test['SubjectID'].unique()
        n_subject = len(subjects_test)
        print(f"n_subject: {n_subject}", len(df_test), len(test_index))
        results = -999 * np.ones((n_subject, n_timepoint))  # longitutinal prediction results

        df_result = pd.DataFrame(results, columns=timepoints, index=subjects_test).copy()
        df_result_shift = pd.DataFrame(results.copy(), columns=timepoints, index=subjects_test).copy()
        df_result_shift_label = pd.DataFrame(results.copy(), columns=timepoints, index=subjects_test).copy()
        for s in (subjects_test):
            df_i = df_test[df_test.SubjectID == s]



            temp_dict = {i: [np.nan] for i in timepoints}
            temp_dict_label = {i: [np.nan] for i in timepoints}

            for t in df_i.VISCODE.values:
                temp_dict[t] = [df_i[df_i.VISCODE == t].stage_pred.values[0]]
                temp_dict_label[t] = [df_i[df_i.VISCODE == t].stage.values[0]]
            predict_timepoint = [v[0] for k, v in temp_dict.items()]
            predict_timepoint_label = [v[0] for k, v in temp_dict_label.items()]
            while np.isnan(predict_timepoint[0]):
                predict_timepoint = predict_timepoint[1:] + [predict_timepoint[0]]
                predict_timepoint_label = predict_timepoint_label[1:] + [predict_timepoint_label[0]]

                # assign to df_result_shift
            df_result_shift.loc[s, :] = predict_timepoint
            # print(df_result_shift)
            df_result_shift_label.loc[s, :] = predict_timepoint_label

        print(f"====================adding {len(df_result_shift)}====================")
        if len(df_result_shift_repreated) == 0:
            df_result_shift_repreated = df_result_shift
            df_result_shift_label_repreated = df_result_shift_label
            # df_result_repeated = df_result
        else:
            df_result_shift_repreated = pd.concat([df_result_shift_repreated, df_result_shift], axis=0)
            df_result_shift_label_repreated = pd.concat([df_result_shift_label_repreated, df_result_shift_label], axis=0)

    return df_result_shift_repreated, df_result_shift_label_repreated