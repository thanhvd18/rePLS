import os
import sys



import numpy as np
import pandas as pd
from rePLS import rePLS, rePCR, reMLR
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline,make_pipeline
from typing import Tuple, List, Dict, Any, Optional
from cross_validator import CrossValidator
from icecream import ic
from sklearn.utils import resample
from tqdm import tqdm
from sklearn.model_selection import train_test_split


import figures.figure1 as fig1
import figures.figure3 as fig3
from figures.figure3.utils import cal_correlation_MSE_regression

def stratified_train_test_df_split(df, test_size,random_state):
    df, selected_subjects, labels = fig1.utils.categorize_disease_group(df)
    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    outcomes = Y.columns.values
    confounders = Z.columns.values
    n_outcomes = Y.shape[1]

    subjects_train, subjects_test, labels_train, labels_test = train_test_split(selected_subjects,labels,test_size=test_size, random_state=random_state )
    train_df = df[df['SubjectID'].isin(subjects_train)]
    test_df = df[df['SubjectID'].isin(subjects_test)]
    N_subject_train = len(subjects_train)
    return train_df, test_df,N_subject_train


def k_fold_prediction_PQ(df: pd.DataFrame,cv: CrossValidator,out_dir: str,n_components: int,random_state: int,n_splits: int,method) -> Tuple[np.ndarray, np.ndarray]:
    df, selected_subjects, labels = fig1.utils.categorize_disease_group(df)
    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    selected_subjects = df['SubjectID'].unique()
    outcomes = Y.columns.values
    confounders = Z.columns.values
    n_outcomes = Y.shape[1]

    stat_df = pd.DataFrame(columns=["fold", "r", "MSE", "p_value"])
    predict_result_df = pd.DataFrame(columns=["outcome" + str(i) for i in range(
        n_outcomes)] + ["outcome" + str(i) + "_pred" for i in range(n_outcomes)] + ["DX_encode"])
    df['DX_encode'] = df['DX'].map({'CN': 0, 'MCI': 1, 'AD': 2})

    Ps = []
    PQs = []
    alphas = []
    for fold, (train_index, test_index) in enumerate(cv.get_splits(selected_subjects,labels)):
        print(f"Processing fold {fold}")
        train_subjects = np.array(selected_subjects)[train_index]
        test_subjects = np.array(selected_subjects)[test_index]

        df_train = df[df['SubjectID'].isin(train_subjects)]
        df_test = df[df['SubjectID'].isin(test_subjects)]

        X_train = np.vstack(df_train['Schaefer_200_7'].apply(eval))
        Y_train = np.array(df_train[outcomes])
        Z_train = np.array(df_train[confounders], dtype=float)

        X_test = np.vstack(df_test['Schaefer_200_7'].apply(eval))
        Y_test = np.array(df_test[outcomes])
        Y_test_ = np.copy(Y_test)
        Z_test = np.array(df_test[confounders], dtype=float)

        X_train, X_test = X_train[:, 2:], X_test[:, 2:]

        # normalize X, Y scaler
        X_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        Y_scaler = StandardScaler()
        Y_train = Y_scaler.fit_transform(Y_train)
        Y_test = Y_scaler.transform(Y_test)

        if method == "PLS":
            model = PLSRegression(n_components=n_components)
            model.fit(X_train, Y_train)
            y_pred = np.array(model.predict(X_test))
            P = model.x_loadings_
            PQ = model.coef_
        elif method == "rePLS":
            model = rePLS(n_components=n_components, Z=Z_train)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test, Z=Z_test)
            P = model.P
            Q = model.Q
            PQ = model.PQ
        else:
            # raise
            print("Method not supported")
            raise ValueError("Method not supported")

        Ps.append(P)
        PQs.append(PQ)
    mean_P = np.mean(Ps, axis=0)
    mean_PQ = np.mean(PQs, axis=0)
    return mean_PQ, mean_P


def k_fold_prediction(df: pd.DataFrame,cv: CrossValidator,out_dir: str,n_components: int,random_state: int,n_splits: int) -> Tuple[np.ndarray, np.ndarray]:
    df, selected_subjects, labels = fig1.utils.categorize_disease_group(df)
    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    selected_subjects = df['SubjectID'].unique()
    outcomes = Y.columns.values
    confounders = Z.columns.values
    n_outcomes = Y.shape[1]


    df['DX_encode'] = df['DX'].map({'CN': 0, 'MCI': 1, 'AD': 2})
    Ps = []
    Y_test_all_fold = []
    Y_pred_all_fold = []
    for fold, (train_index, test_index) in enumerate(cv.get_splits(selected_subjects,labels)):
        print(f"Processing fold {fold}")
        train_subjects = np.array(selected_subjects)[train_index]
        test_subjects = np.array(selected_subjects)[test_index]

        df_train = df[df['SubjectID'].isin(train_subjects)]
        df_test = df[df['SubjectID'].isin(test_subjects)]

        X_train = np.vstack(df_train['Schaefer_200_7'].apply(eval))
        Y_train = np.array(df_train[outcomes])
        Z_train = np.array(df_train[confounders], dtype=float)

        X_test = np.vstack(df_test['Schaefer_200_7'].apply(eval))
        Y_test = np.array(df_test[outcomes])
        Z_test = np.array(df_test[confounders], dtype=float)

        X_train, X_test = X_train[:, :], X_test[:, :]

        X_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        # Y_scaler = StandardScaler()
        # Y_train = Y_scaler.fit_transform(Y_train)
        # Y_test = Y_scaler.transform(Y_test)

        model = rePLS(n_components=n_components, Z=Z_train)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test, Z=Z_test)

        Y_test_all_fold.append(Y_test)
        Y_pred_all_fold.append(y_pred)

        Ps.append(model.P)

    scores = {}
    r, MSE, p_value = cal_correlation_MSE_regression(Y_test, y_pred)
    scores["r"] = r
    scores["MSE"] = MSE
    scores["p_value"] = p_value

    mean_P = np.mean(Ps, axis=0)
    return mean_P, scores


def sample_size_change_prediction(df: pd.DataFrame,cv: CrossValidator,out_dir: str,n_components: int,random_state: int,n_splits: int)-> pd.DataFrame:
    df, selected_subjects, labels = fig1.utils.categorize_disease_group(df)
    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    selected_subjects = df['SubjectID'].unique()
    outcomes = Y.columns.values
    confounders = Z.columns.values
    n_outcomes = Y.shape[1]


    df['DX_encode'] = df['DX'].map({'CN': 0, 'MCI': 1, 'AD': 2})


    result_df = pd.DataFrame(columns=['perc', 'method', 'r', 'mean_MSE', 'pvalue', 'var_MSE', 'var_r'])
    percentages = [0.1, 0.3, 0.5, 0.7, 0.9, 1]

    for perc in tqdm(percentages):
        print(f"Processing percentage {perc}")


        methods = ['rePLS', 'rePCR', 'reMLR', 'PLS', 'PCR', 'LR']
        r_in_kfold = { method: [] for method in methods}
        MSE_in_kfold = { method: [] for method in methods}
        pvalue_in_kfold = { method: [] for method in methods}
        for fold, (train_index, test_index) in enumerate(cv.get_splits(selected_subjects, labels)):
            print(f"Processing fold {fold}")
            N = len(train_index)
            sample_size = int(perc * N)
            train_subjects_idx = resample(train_index, replace=False, n_samples=sample_size, random_state=random_state)
            train_subjects = np.array(selected_subjects)[train_subjects_idx]

            test_subjects = np.array(selected_subjects)[test_index]

            df_train = df[df['SubjectID'].isin(train_subjects)]
            df_test = df[df['SubjectID'].isin(test_subjects)]

            X_train = np.vstack(df_train['Schaefer_200_7'].apply(eval))
            Y_train = np.array(df_train[outcomes])
            Z_train = np.array(df_train[confounders], dtype=float)

            X_test = np.vstack(df_test['Schaefer_200_7'].apply(eval))
            Y_test = np.array(df_test[outcomes])
            Y_test_ = np.copy(Y_test)
            Z_test = np.array(df_test[confounders], dtype=float)

            X_train, X_test = X_train[:, :], X_test[:, :]

            X_scaler = StandardScaler()
            X_train = X_scaler.fit_transform(X_train)
            X_test = X_scaler.transform(X_test)

            Y_scaler = StandardScaler()
            Y_train = Y_scaler.fit_transform(Y_train)
            Y_test = Y_scaler.transform(Y_test)


            for method in methods:
                if method == "LR":
                    model = LinearRegression()
                    model.fit(X_train, Y_train)
                    y_pred = np.array(model.predict(X_test))
                elif method == "reMLR":
                    model = reMLR(Z=Z_train)
                    model.fit(X_train, Y_train)
                    y_pred = model.predict(X_test, Z=Z_test)
                elif method == "PCR":
                    model = make_pipeline(PCA(n_components=n_components), LinearRegression())
                    model.fit(X_train, Y_train)
                    # Predict on test set
                    y_pred = model.predict(X_test)
                elif method == "rePCR":
                    model = rePCR(n_components=n_components, Z=Z_train)
                    model.fit(X_train, Y_train)
                    y_pred = model.predict(X_test, Z=Z_test)

                elif method == "PLS":
                    model = PLSRegression(n_components=n_components)
                    model.fit(X_train, Y_train)
                    y_pred = np.array(model.predict(X_test))
                elif method == "rePLS":
                    model = rePLS(n_components=n_components, Z=Z_train)
                    model.fit(X_train, Y_train)
                    y_pred = model.predict(X_test, Z=Z_test)

                else:
                    print("Method not supported")
                    raise ValueError("Method not supported")

                r, MSE, p_value = cal_correlation_MSE_regression(Y_test, y_pred)
                r_in_kfold[method].append(r)
                MSE_in_kfold[method].append(MSE)
                pvalue_in_kfold[method].append(p_value)



        result_df.loc[len(result_df)] = [perc, 'rePLS', np.mean(r_in_kfold['rePLS']), np.mean(MSE_in_kfold['rePLS']), np.mean(pvalue_in_kfold['rePLS']), np.var(MSE_in_kfold['rePLS']), np.var(r_in_kfold['rePLS'])]
        result_df.loc[len(result_df)] = [perc, 'rePCR', np.mean(r_in_kfold['rePCR']), np.mean(MSE_in_kfold['rePCR']), np.mean(pvalue_in_kfold['rePCR']), np.var(MSE_in_kfold['rePCR']), np.var(r_in_kfold['rePCR'])]
        result_df.loc[len(result_df)] = [perc, 'reMLR', np.mean(r_in_kfold['reMLR']), np.mean(MSE_in_kfold['reMLR']), np.mean(pvalue_in_kfold['reMLR']), np.var(MSE_in_kfold['reMLR']), np.var(r_in_kfold['reMLR'])]
        result_df.loc[len(result_df)] = [perc, 'PLS', np.mean(r_in_kfold['PLS']), np.mean(MSE_in_kfold['PLS']), np.mean(pvalue_in_kfold['PLS']), np.var(MSE_in_kfold['PLS']), np.var(r_in_kfold['PLS'])]
        result_df.loc[len(result_df)] = [perc, 'PCR', np.mean(r_in_kfold['PCR']), np.mean(MSE_in_kfold['PCR']), np.mean(pvalue_in_kfold['PCR']), np.var(MSE_in_kfold['PCR']), np.var(r_in_kfold['PCR'])]
        result_df.loc[len(result_df)] = [perc, 'LR', np.mean(r_in_kfold['LR']), np.mean(MSE_in_kfold['LR']), np.mean(pvalue_in_kfold['LR']), np.var(MSE_in_kfold['LR']), np.var(r_in_kfold['LR'])]

    os.makedirs(out_dir, exist_ok=True)
    result_df.to_csv(os.path.join(out_dir, 'sample_size_change_prediction.csv'))
    print("Save to ", os.path.join(out_dir, 'sample_size_change_prediction.csv'))
    return result_df


def number_of_outcomes_prediction(df: pd.DataFrame,cv: CrossValidator,out_dir: str,n_components: int,random_state: int,n_splits: int)-> pd.DataFrame:
    df, selected_subjects, labels = fig1.utils.categorize_disease_group(df)
    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    selected_subjects = df['SubjectID'].unique()
    outcomes = Y.columns.values
    confounders = Z.columns.values
    n_outcomes = Y.shape[1]


    df['DX_encode'] = df['DX'].map({'CN': 0, 'MCI': 1, 'AD': 2})
    import itertools
    outcome_combination_df = pd.DataFrame(columns=['n.outcome', 'combine'])
    for n_outcome in range(1, n_outcomes + 1):
        print(f"Processing number of outcomes {n_outcome}")
        outcome_combinations = list(itertools.combinations(outcomes, n_outcome))
        for outcome_combination in outcome_combinations:
            outcome_combination_df.loc[len(outcome_combination_df)] = [n_outcome, outcome_combination]
    # ic(outcome_combination_df)


    result_df = pd.DataFrame(columns=['n.outcome', 'combination', 'r', 'mean_MSE', 'pvalue', 'var_MSE', 'var_r'])



    for row in outcome_combination_df.iterrows():
        n_outcome = row[1]['n.outcome']
        outcome_combination = list(row[1]['combine'])

        print(f"Processing number of outcomes {n_outcome} with combination {outcome_combination}")


        r_in_kfold = []
        MSE_in_kfold = []
        pvalue_in_kfold = []
        for fold, (train_index, test_index) in enumerate(cv.get_splits(selected_subjects, labels)):
            print(f"Processing fold {fold}")

            train_subjects = np.array(selected_subjects)[train_index]

            test_subjects = np.array(selected_subjects)[test_index]

            df_train = df[df['SubjectID'].isin(train_subjects)]
            df_test = df[df['SubjectID'].isin(test_subjects)]

            X_train = np.vstack(df_train['Schaefer_200_7'].apply(eval))
            Y_train = np.array(df_train[outcome_combination])
            Z_train = np.array(df_train[confounders], dtype=float)

            X_test = np.vstack(df_test['Schaefer_200_7'].apply(eval))
            Y_test = np.array(df_test[outcome_combination])
            Y_test_ = np.copy(Y_test)
            Z_test = np.array(df_test[confounders], dtype=float)

            X_train, X_test = X_train[:, :], X_test[:, :]

            # normalize X, Y scaler
            X_scaler = StandardScaler()
            X_train = X_scaler.fit_transform(X_train)
            X_test = X_scaler.transform(X_test)

            Y_scaler = StandardScaler()
            Y_train = Y_scaler.fit_transform(Y_train)
            Y_test = Y_scaler.transform(Y_test)



            model = rePLS(n_components=n_components, Z=Z_train)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test, Z=Z_test)



            r, MSE, p_value = cal_correlation_MSE_regression(Y_test, y_pred)
            r_in_kfold.append(r)
            MSE_in_kfold.append(MSE)
            pvalue_in_kfold.append(p_value)

        result_df.loc[len(result_df)] = [n_outcome, outcome_combination, np.mean(r_in_kfold), np.mean(MSE_in_kfold), np.mean(pvalue_in_kfold), np.var(MSE_in_kfold), np.var(r_in_kfold)]



    #save to out_dir
    print(result_df)
    os.makedirs(out_dir, exist_ok=True)
    result_df.to_csv(os.path.join(out_dir, 'n_outcome_change_prediction.csv'))
    return result_df


