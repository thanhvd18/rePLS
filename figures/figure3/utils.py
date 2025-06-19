import os
import sys
# sys.path.append((os.path.join(os.getcwd(), '..')))
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline,make_pipeline

sys.path.append(os.path.join(os.getcwd(), '..', 'dev'))

from cross_validator import CrossValidator
from typing import Tuple, List, Dict, Any, Optional

from rePLS import rePLS, rePCR, reMLR
import figures.figure1 as fig1


def cal_correlation_MSE_regression(y_test: np.ndarray, y_pred: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
    rs = []
    MSEs = []
    p_values = []
    # for each outcome
    for i in range(y_test.shape[1]):
        r, p = pearsonr(
            np.array(y_test[:, i]).ravel(), np.array(y_pred[:, i]).ravel())
        rs.append(r)
        p_values.append(p)
        MSEs.append(mean_squared_error(y_test[:, i], y_pred[:, i]))
    return rs, MSEs, p_values

def k_fold_prediction(df: pd.DataFrame,cv: CrossValidator,out_dir: str,method: str,n_components: int,random_state: int,n_splits: int, dataset: str = "ADNI") -> List[pd.DataFrame]:
    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    selected_subjects = df['SubjectID'].unique()
    outcomes = Y.columns.values
    confounders = Z.columns.values
    n_outcomes = Y.shape[1]

    stat_df = pd.DataFrame(columns=["fold", "r", "MSE", "p_value"])
    predict_result_df = pd.DataFrame(columns=["outcome" + str(i) for i in range(
        n_outcomes)] + ["outcome" + str(i) + "_pred" for i in range(n_outcomes)] + ["DX_encode"] + ["idx"])
    df['DX_encode'] = df['DX'].map({'CN': 0, 'MCI': 1, 'AD': 2})
    for fold, (train_index, test_index) in enumerate(cv.get_splits(selected_subjects)):
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

        X_train, X_test = X_train[:, :], X_test[:, :]

        # normalize X, Y scaler
        X_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        Y_scaler = StandardScaler()
        Y_train = Y_scaler.fit_transform(Y_train)
        Y_test = Y_scaler.transform(Y_test)

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
            # raise
            print("Method not supported")
            raise ValueError("Method not supported")

        r, MSE, p_value = cal_correlation_MSE_regression(Y_test, y_pred)

        stat_df = pd.concat([stat_df, pd.DataFrame(
            [{"fold": fold, "r": r, "MSE": MSE, "p_value": p_value}])])
        if fold == 0:
            predict_result_df[[
                f"outcome{i}" for i in range(len(outcomes))]] = (Y_test_)
            predict_result_df[[
                f"outcome{i}_pred" for i in range(len(outcomes))]] = Y_scaler.inverse_transform(y_pred)
            predict_result_df["DX_encode"] = df[df['SubjectID'].isin(test_subjects)].DX_encode.values.copy()
            predict_result_df["idx"] = df_test.index.values
        else:
            fold_i_prediction_results_df = pd.DataFrame(columns=["outcome" + str(i) for i in range(
                n_outcomes)] + ["outcome" + str(i) + "_pred" for i in range(n_outcomes)] + ["DX_encode"])
            fold_i_prediction_results_df[[
                "outcome" + str(i) for i in range(n_outcomes)]] = (Y_test_)

            fold_i_prediction_results_df[[
                "outcome" + str(i) + "_pred" for i in range(n_outcomes)]] = Y_scaler.inverse_transform(y_pred)
            fold_i_prediction_results_df["DX_encode"] = df[df['SubjectID'].isin(test_subjects)].DX_encode.values.copy()
            fold_i_prediction_results_df["idx"] = df_test.index.values

            predict_result_df = pd.concat(
                [predict_result_df, fold_i_prediction_results_df])

    # Initialize DataFrame to store statistics
    combine_stat_df = pd.DataFrame(columns=["outcome", "r", "MSE", "p_value"])

    # Loop through outcomes
    for i in range(n_outcomes):
        outcome_i = predict_result_df[f"outcome{i}"]
        pred_outcome_i = predict_result_df[f"outcome{i}_pred"]

        # Calculate r and MSE
        correlation, p_value = pearsonr(outcome_i, pred_outcome_i)
        mse = mean_squared_error(outcome_i, pred_outcome_i)

        # Log results
        print(f"Outcome: {outcomes[i]}, Correlation: {correlation}, p_value: {p_value}, MSE: {mse}")

        # Append to DataFrame
        combine_stat_df.loc[i] = {"outcome": outcomes[i], "r": correlation, "MSE": mse, "p_value": p_value}

    # Define filenames
    stat_filename = f"stat_t{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}.csv"
    predict_filename = f"predict_{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}.csv"
    combine_stat_filename = f"combine_stat_{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}.csv"

    # Get absolute paths
    stat_path = os.path.abspath(os.path.join(out_dir, stat_filename))
    predict_path = os.path.abspath(os.path.join(out_dir, predict_filename))
    combine_stat_path = os.path.abspath(os.path.join(out_dir, combine_stat_filename))

    # Save results to CSV
    os.makedirs(out_dir, exist_ok=True)
    stat_df.to_csv(stat_path, index=False)
    predict_result_df.sort_values(by="DX_encode", inplace=True)
    predict_result_df.to_csv(predict_path, index=False)
    combine_stat_df.to_csv(combine_stat_path, index=False)

    # Print absolute paths
    print(f"Stat file saved to: {stat_path}")
    print(f"Predict file saved to: {predict_path}")
    print(f"Combine stat file saved to: {combine_stat_path}")
    return stat_df,predict_result_df,combine_stat_df






