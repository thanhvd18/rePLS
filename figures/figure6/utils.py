

import pandas as pd
import numpy as np
from fontTools.misc.arrayTools import scaleRect
from sklearn.preprocessing import LabelEncoder
import os
import sys


sys.path.append((os.path.join(os.getcwd(), 'figure1')))
import figure1 as fig1
import figure3 as fig3
from figure3.utils import cal_correlation_MSE_regression

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import scipy

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
from scipy.linalg import pinv as pinv2

def load_OASIS(data_path: str) -> pd.DataFrame:
    """
        Extracts and filters data from the OASIS dataset.

        This function reads a dataset from the specified file path into a pandas DataFrame.
        It filters rows containing NaN values in the 'MMSE' and 'CDR' columns and
        returns the resulting filtered DataFrame.

        Parameters:
        -----------
        data_path : str
            The file path of the dataset to be loaded. The dataset must be
            in CSV format.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing the filtered data where rows with NaN values
            in the 'MMSE' or 'CDR' columns are removed.
    """
    merged_df = pd.read_csv(data_path)
    merged_df_filter = merged_df[(~merged_df["MMSE"].isna()) & (~merged_df["CDR"].isna())]
    return merged_df_filter

def get_input_output_confounder_OASIS(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_encoder = LabelEncoder()
    X = df["thickness"].apply(lambda x: np.array(eval(x)))
    X = np.vstack(X)

    # scaler normalization
    Y = df[["CDR", "MMSE"]].values
    # Y = scaler_y.fit_transform(Y)
    # print(X.shape, Y.shape)

    # Fit and transform the 'category' column
    df['M/F_encoded'] = label_encoder.fit_transform(df['M/F'])
    Z = df[["M/F_encoded", "Age"]].values
    return X, Y, Z

def k_fold_prediction_OASIS(df: pd.DataFrame,cv: CrossValidator,out_dir: str,method: str,n_components: int,random_state: int,n_splits: int, dataset: str = "OASIS") -> List[pd.DataFrame]:
    X, Y, Z = get_input_output_confounder_OASIS(df)
    n_outcomes = Y.shape[1]
    selected_subjects = list(range(X.shape[0]))
    stat_df = pd.DataFrame(columns=["fold", "r", "MSE", "p_value"])
    predict_result_df = pd.DataFrame(columns=["outcome" + str(i) for i in range(
        n_outcomes)] + ["outcome" + str(i) + "_pred" for i in range(n_outcomes)] + ["DX_encode"])

    for fold, (train_index, test_index) in enumerate(cv.get_splits(selected_subjects)):
        print(f"Processing fold {fold}")


        X_train = X[train_index,:]
        Y_train = Y[train_index,:]
        Z_train = Z[train_index,:]

        X_test = X[test_index,:]
        Y_test = Y[test_index,:]
        Y_test_ = np.copy(Y_test)
        Z_test = Z[test_index,:]


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
                f"outcome{i}" for i in range(n_outcomes)]] = (Y_test_)
            predict_result_df[[
                f"outcome{i}_pred" for i in range(n_outcomes)]] = Y_scaler.inverse_transform(y_pred)
        else:
            fold_i_prediction_results_df = pd.DataFrame(columns=["outcome" + str(i) for i in range(
                n_outcomes)] + ["outcome" + str(i) + "_pred" for i in range(n_outcomes)] + ["DX_encode"])
            fold_i_prediction_results_df[[
                "outcome" + str(i) for i in range(n_outcomes)]] = (Y_test_)

            fold_i_prediction_results_df[[
                "outcome" + str(i) + "_pred" for i in range(n_outcomes)]] = Y_scaler.inverse_transform(y_pred)

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

        # Append to DataFrame
        combine_stat_df.loc[i] = {"outcome": i, "r": correlation, "MSE": mse, "p_value": p_value}

    # Define filenames
    stat_filename = f"stat_t{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}.csv"
    predict_filename = f"predict_{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}.csv"
    combine_stat_filename = f"combine_stat_{dataset}_n_splits{n_splits}_method_{method}_n_components{n_components}_random_state{random_state}.csv"

    # Get absolute paths
    os.makedirs(out_dir, exist_ok=True)
    stat_path = os.path.abspath(os.path.join(out_dir, stat_filename))
    predict_path = os.path.abspath(os.path.join(out_dir, predict_filename))
    combine_stat_path = os.path.abspath(os.path.join(out_dir, combine_stat_filename))

    # Save results to CSV

    stat_df.to_csv(stat_path, index=False)
    predict_result_df.sort_values(by="DX_encode", inplace=True)
    predict_result_df.to_csv(predict_path, index=False)
    combine_stat_df.to_csv(combine_stat_path, index=False)

    # Print absolute paths
    print(f"Stat file saved to: {stat_path}")
    print(f"Predict file saved to: {predict_path}")
    print(f"Combine stat file saved to: {combine_stat_path}")
    return stat_df,predict_result_df,combine_stat_df

def k_fold_train_OASIS(df: pd.DataFrame,cv: CrossValidator,out_dir: str,method: str,n_components: int,random_state: int,n_splits: int, dataset: str = "OASIS") -> List[pd.DataFrame]:
    X, Y, Z = get_input_output_confounder_OASIS(df)
    n_outcomes = Y.shape[1]
    selected_subjects = list(range(X.shape[0]))
    stat_df = pd.DataFrame(columns=["fold", "r", "MSE", "p_value"])
    predict_result_df = pd.DataFrame(columns=["outcome" + str(i) for i in range(
        n_outcomes)] + ["outcome" + str(i) + "_pred" for i in range(n_outcomes)] + ["DX_encode"])

    models = []
    scaler_Xs = []
    scaler_Ys = []
    for fold, (train_index, test_index) in enumerate(cv.get_splits(selected_subjects)):
        print(f"Processing fold {fold}")


        X_train = X[train_index,:]
        Y_train = Y[train_index,:]
        Z_train = Z[train_index,:]

        X_test = X[test_index,:]
        Y_test = Y[test_index,:]
        Y_test_ = np.copy(Y_test)
        Z_test = Z[test_index,:]


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

        models.append(model)
        scaler_Xs.append(X_scaler)
        scaler_Ys.append(Y_scaler)
    return models, scaler_Xs, scaler_Ys




def k_fold_train_ADNI(df: pd.DataFrame,cv: CrossValidator,out_dir: str,method: str,n_components: int,random_state: int,n_splits: int, dataset: str = "ADNI") -> List[pd.DataFrame]:
    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    selected_subjects = df['SubjectID'].unique()
    outcomes = Y.columns.values
    confounders = Z.columns.values
    n_outcomes = Y.shape[1]

    stat_df = pd.DataFrame(columns=["fold", "r", "MSE", "p_value"])
    predict_result_df = pd.DataFrame(columns=["outcome" + str(i) for i in range(
        n_outcomes)] + ["outcome" + str(i) + "_pred" for i in range(n_outcomes)] + ["DX_encode"])
    df['DX_encode'] = df['DX'].map({'CN': 0, 'MCI': 1, 'AD': 2})

    models = []
    scaler_Xs = []
    scaler_Ys = []

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
        scale_X = StandardScaler()
        X_train = scale_X.fit_transform(X_train)
        X_test = scale_X.transform(X_test)

        scaler_Y = StandardScaler()
        Y_train = scaler_Y.fit_transform(Y_train)
        Y_test = scaler_Y.transform(Y_test)

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

        models.append(model)
        scaler_Xs.append(scale_X)
        scaler_Ys.append(scaler_Y)


    return models, scaler_Xs, scaler_Ys

def min_max_normalize(P, thresh_per=0.8):
    P_std = (P - P.min()) / (P.max() - P.min())
    P_scaled = P_std * 2 - 1
    vol = np.sort(P_scaled.flatten())
    min_thresh = vol[int(len(vol) * thresh_per)]
    max_thresh = vol[int(len(vol) * (1 - thresh_per))]
    P_thresh = np.zeros_like(P_scaled)
    P_thresh[P_scaled >= max_thresh] = 1
    P_thresh[P_scaled <= min_thresh] = -1
    return P_thresh

def consistent_brainmap_k_fold_repeated_OASIS(df: pd.DataFrame,cv: CrossValidator,out_dir: str,method: str,n_components: int,random_state: int,n_splits: int,  thresh_per: float,intersection_perc: float) -> List[str]:
    X, Y, Z = get_input_output_confounder_OASIS(df)
    n_outcomes = Y.shape[1]
    selected_subjects = list(range(X.shape[0]))
    stat_df = pd.DataFrame(columns=["fold", "r", "MSE", "p_value"])
    predict_result_df = pd.DataFrame(columns=["outcome" + str(i) for i in range(
        n_outcomes)] + ["outcome" + str(i) + "_pred" for i in range(n_outcomes)] + ["DX_encode"])

    n_features = X.shape[1]
    outcomes = ["CDR", "MMSE"]



    PQ_union = np.zeros([n_features, len(outcomes)])  # Assuming 200 features

    P_union = np.zeros([n_features, n_components])  # Assuming 200 features
    Ps = []
    for fold, (train_index, test_index) in enumerate(cv.get_splits(selected_subjects)):
        print(f"Processing fold {fold}")


        X_train = X[train_index,:]
        Y_train = Y[train_index,:]
        Z_train = Z[train_index,:]

        X_test = X[test_index,:]
        Y_test = Y[test_index,:]
        Y_test_ = np.copy(Y_test)
        Z_test = Z[test_index,:]


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
        elif method == "rePLS":
            model = rePLS(n_components=n_components, Z=Z_train)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test, Z=Z_test)

            PQ = model.PQ

            P = model.P
            Ps.append(P)


        else:
            # raise
            print("Method not supported")
            raise ValueError("Method not supported")

        PQ_thresh = np.zeros_like(PQ)
        P_thresh = np.zeros_like(P)
        for i in range(n_outcomes):
            PQ_thresh[:, i] = min_max_normalize(
                PQ[:, i], thresh_per=thresh_per)
            PQ_thresh[:, i] = np.where(
                PQ_thresh[:, i] > 0, 1, PQ_thresh[:, i])
            PQ_thresh[:, i] = np.where(
                PQ_thresh[:, i] < 0, -1, PQ_thresh[:, i])
        PQ_union += PQ_thresh

        for i in range(n_components):
            P_thresh[:, i] = min_max_normalize(
                P[:, i], thresh_per=thresh_per)
            P_thresh[:, i] = np.where(
                P_thresh[:, i] > 0, 1, P_thresh[:, i])
            P_thresh[:, i] = np.where(
                P_thresh[:, i] < 0, -1, P_thresh[:, i])
        P_union += P_thresh
    repeated_time = fold+1
    mean_P = np.mean(Ps, axis=0)
    # save mean P
    scipy.io.savemat(os.path.join(out_dir, f'mean_P_OASIS_{method}_n_splits{n_splits}_n_repeats{repeated_time}.mat'), {"mean_P": mean_P})
        # Calculate intersections
    PQ_intersection1 = (PQ_union >= repeated_time *
                        intersection_perc).astype(int)
    PQ_intersection2 = (PQ_union <= -repeated_time *
                        intersection_perc).astype(int)
    PQ_intersection = PQ_intersection1 + PQ_intersection2
    # print("Intersection result:", PQ_intersection)

    os.makedirs(out_dir, exist_ok=True)
    PQ_path = os.path.join(out_dir,
                           f'PQ_intersection_OASIS_thresh_per{thresh_per}_intersection_perc{intersection_perc}_n_splits{n_splits}_n_repeats{repeated_time}.csv')
    pd.DataFrame(PQ_intersection).to_csv(PQ_path,
                                         index=False, header=outcomes)

    P_intersection1 = (P_union >= repeated_time *
                       intersection_perc).astype(int)
    P_intersection2 = (P_union <= -repeated_time *
                       intersection_perc).astype(int)
    P_intersection = P_intersection1 + P_intersection2
    # print("Intersection result:", P_intersection)
    P_path = os.path.join(out_dir,
                          f'P_intersection_OASIS_thresh_per{thresh_per}_intersection_perc{intersection_perc}_n_splits{n_splits}_n_repeats{repeated_time}.csv')
    pd.DataFrame(P_intersection).to_csv(
        P_path,
        index=False, header=[f'Component {i}' for i in range(n_components)])
    print(f"Saved PQ to {PQ_path}")
    print(f"Saved P to {P_path}")
    return PQ_path, P_path




def consistent_brainmap_k_fold_repeated_ADNI(df: pd.DataFrame,cv: CrossValidator,out_dir: str,method: str,n_components: int,random_state: int,n_splits: int,  thresh_per: float,intersection_perc: float) -> List[str]:
    X, Y, Z = fig1.utils.get_input_output_confounder(df)
    selected_subjects = df['SubjectID'].unique()
    outcomes = Y.columns.values
    confounders = Z.columns.values
    n_outcomes = Y.shape[1]
    n_features = X.shape[1]


    PQ_union = np.zeros([n_features, len(outcomes)])  # Assuming 200 features

    P_union = np.zeros([n_features, n_components])  # Assuming 200 features
    Ps = []
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
        scale_X = StandardScaler()
        X_train = scale_X.fit_transform(X_train)
        X_test = scale_X.transform(X_test)

        scaler_Y = StandardScaler()
        Y_train = scaler_Y.fit_transform(Y_train)
        Y_test = scaler_Y.transform(Y_test)


        if method == "PLS":
            model = PLSRegression(n_components=n_components)
            model.fit(X_train, Y_train)
            y_pred = np.array(model.predict(X_test))
            P = model.x_loadings_
            PQ = model.y_scores_.T
            Ps.append(P)
        elif method == "rePLS":
            model = rePLS(n_components=n_components, Z=Z_train)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test, Z=Z_test)
            PQ = model.PQ

            P = model.P
            Ps.append(P)

        else:
            # raise
            print("Method not supported")
            raise ValueError("Method not supported")
        PQ_thresh = np.zeros_like(PQ)
        P_thresh = np.zeros_like(P)
        for i in range(n_outcomes):
            PQ_thresh[:, i] = min_max_normalize(
                PQ[:, i], thresh_per=thresh_per)
            PQ_thresh[:, i] = np.where(
                PQ_thresh[:, i] > 0, 1, PQ_thresh[:, i])
            PQ_thresh[:, i] = np.where(
                PQ_thresh[:, i] < 0, -1, PQ_thresh[:, i])
        PQ_union += PQ_thresh

        for i in range(n_components):
            P_thresh[:, i] = min_max_normalize(
                P[:, i], thresh_per=thresh_per)
            P_thresh[:, i] = np.where(
                P_thresh[:, i] > 0, 1, P_thresh[:, i])
            P_thresh[:, i] = np.where(
                P_thresh[:, i] < 0, -1, P_thresh[:, i])
        P_union += P_thresh
    mean_P = np.mean(Ps, axis=0)
    # save mean P
    scipy.io.savemat(os.path.join(out_dir, f'mean_P_ADNI_{method}_n_splits{n_splits}_n_repeats{fold + 1}.mat'),
                     {'mean_P': mean_P})

    repeated_time = fold+1
    # Calculate intersections
    PQ_intersection1 = (PQ_union >= repeated_time *
                        intersection_perc).astype(int)
    PQ_intersection2 = (PQ_union <= -repeated_time *
                        intersection_perc).astype(int)
    PQ_intersection = PQ_intersection1 + PQ_intersection2
    # print("Intersection result:", PQ_intersection)

    os.makedirs(out_dir, exist_ok=True)
    PQ_path = os.path.join(out_dir, f'PQ_intersection_ADNI_thresh_per{thresh_per}_intersection_perc{intersection_perc}_n_splits{n_splits}_n_repeats{repeated_time}.csv')
    pd.DataFrame(PQ_intersection).to_csv(PQ_path,
        index=False, header=outcomes)

    P_intersection1 = (P_union >= repeated_time *
                       intersection_perc).astype(int)
    P_intersection2 = (P_union <= -repeated_time *
                       intersection_perc).astype(int)
    P_intersection = P_intersection1 + P_intersection2
    # print("Intersection result:", P_intersection)
    P_path = os.path.join(out_dir, f'P_intersection_ADNI_thresh_per{thresh_per}_intersection_perc{intersection_perc}_n_splits{n_splits}_n_repeats{repeated_time}.csv')
    pd.DataFrame(P_intersection).to_csv(
        P_path,
        index=False, header=[f'Component {i}' for i in range(n_components)])
    print(f"Saved PQ to {PQ_path}")
    print(f"Saved P to {P_path}")
    return PQ_path, P_path

def predict_with_components_PLS(self, X, components=None):
    """
    Predict using a specified number of components from the residual model.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
        Input data for prediction.
    - components: list, optional
        List of components to use for prediction. If None, all components are used.
    Returns:
    - preds: array-like, shape (n_samples,)
        Predictions based on the specified list of components.
    """
    # Zero-centering residuals
    X -= self._x_mean

    # Use only the specified number of components
    if components is not None:
        # if componentse > self.n_components:
        #     raise ValueError(
        #         f"n_components should not exceed the maximum number of components: {self.n_components}"
        #     )

        x_rotations_ = np.dot(
            self.x_weights_[:, components],
            pinv2(np.dot(self.x_loadings_[:, components].T, self.x_weights_[:, components]), check_finite=False),
        )
        # self.y_rotations_ = np.dot(
        #     self.y_weights_[:,n_components],
        #     pinv2(np.dot(self.y_loadings_[:,n_components].T, self.y_weights_[:,n_components]), check_finite=False),
        # )
        coef_ = np.dot(x_rotations_[:, components], self.y_loadings_[:, components].T)
        coef_ = (coef_ * self._y_std).T / self._x_std

        preds_residual = X @ coef_.T + self.intercept_
    else:
        preds_residual = self.predict(X)

    return np.array(preds_residual)

def test_OASIS(models, scaler_Xs, scaler_Ys, X, Y, Z,out_dir,components, method):
    Y_pred_mean = None
    for i, model in enumerate(models):
        scaler_x, scaler_y = scaler_Xs[i], scaler_Ys[i]
        X = scaler_x.fit_transform(X)
        if method == "rePLS":
            Y_pred = model.predict_with_components(X, Z=Z, components=components)
        elif method == "PLS":
            Y_pred = predict_with_components_PLS(model, X, components=components)
        else:
            raise ValueError("Method not supported")
        Y_pred = scaler_y.inverse_transform(Y_pred)
        if i == 0:
            Y_pred_mean = Y_pred
        else:
            Y_pred_mean += Y_pred
    Y_pred_mean /= len(models)
    r, MSE, p_value = cal_correlation_MSE_regression(Y, Y_pred_mean)
    os.makedirs(out_dir, exist_ok=True)
    #save Y, Y_pred_mean
    Y_df = pd.DataFrame( columns=['CDR', 'MMSE', "pred_CDR", "pred_MMSE"])
    Y_df['CDR'] = Y[:,0]
    Y_df['MMSE'] = Y[:,1]
    Y_df['pred_CDR'] = Y_pred_mean[:,0]
    Y_df['pred_MMSE'] = Y_pred_mean[:,4]
    Y_df.to_csv(os.path.join(out_dir, f'prediction_.csv'), index=False)

    stat_df = pd.DataFrame(columns=["r", "MSE", "p_value"])
    stat_df = pd.concat([stat_df, pd.DataFrame(
        [{"r": r, "MSE": MSE, "p_value": p_value}])])
    stat_df.to_csv(os.path.join(out_dir, f'stat_{method}.csv'), index=False)

    return Y_df

def test_ADNI(models, scaler_Xs, scaler_Ys, X, Y, Z,out_dir, method="rePLS"):
    Y_pred_mean = None
    for i, model in enumerate(models):
        scaler_x, scaler_y = scaler_Xs[i], scaler_Ys[i]
        X = scaler_x.fit_transform(X)

        if method == "rePLS" or method == "rePCR" or method == "reMLR":
            Y_pred = model.predict(X, Z=Z)
        else:
            Y_pred = model.predict(X)

        Y_pred = scaler_y.inverse_transform(Y_pred)
        if i == 0:
            Y_pred_mean = Y_pred
        else:
            Y_pred_mean += Y_pred
    Y_pred_mean /= len(models)
    Y = Y.values

    r, MSE, p_value = cal_correlation_MSE_regression(Y[:,[0,4]], Y_pred_mean)
    os.makedirs(out_dir, exist_ok=True)
    #save Y, Y_pred_mean
    Y_df = pd.DataFrame( columns=['CDR', 'MMSE', "pred_CDR", "pred_MMSE"])
    Y_df['CDR'] = Y[:,0]
    Y_df['MMSE'] = Y[:,4]
    Y_df['pred_CDR'] = Y_pred_mean[:,0]
    Y_df['pred_MMSE'] = Y_pred_mean[:,1]
    Y_df.to_csv(os.path.join(out_dir, f'prediction_.csv'), index=False)

    stat_df = pd.DataFrame(columns=["r", "MSE", "p_value"])
    stat_df = pd.concat([stat_df, pd.DataFrame(
    [{"r": r, "MSE": MSE, "p_value": p_value}])])
    stat_df.to_csv(os.path.join(out_dir, f'stat_{method}.csv'), index=False)

    return Y_df


def common_unique_regions(OASIS_P_path,ADNI_P_path, out_dir):
    df_ADNI = pd.read_csv(ADNI_P_path)
    df_OASIS = pd.read_csv( OASIS_P_path)

    ADNI_CDRSB_regions = df_ADNI.iloc[:, 0].values
    OASIS_CDR_regions = df_OASIS.iloc[:, 0].values

    ADNI_MMSE_regions = df_ADNI.iloc[:, 1].values
    OASIS_MMSE_regions = df_OASIS.iloc[:, 1].values


    ADNI_idx = np.where(np.array(ADNI_CDRSB_regions) == 1)[0]
    OASIS_idx = np.where(np.array(OASIS_CDR_regions) == 1)[0]

    common_idx = np.intersect1d(OASIS_idx, ADNI_idx)
    ADNI_union_idx = np.setdiff1d(ADNI_idx, OASIS_idx)
    OASIS_union_idx = np.setdiff1d(OASIS_idx, ADNI_idx)

    weight = np.ones([202, 1]) * -999
    weight[common_idx] = 3
    weight[ADNI_union_idx] = 2
    weight[OASIS_union_idx] = 1

    weight_ = list(weight.T[0])
    # weight_ = [-999, -999] + weight_

    os.makedirs(out_dir, exist_ok=True)
    common_unique_ADNI_OASIS_CDR = weight_
    CDR_file_path = os.path.join(out_dir, f'common_unique_ADNI_OASIS_CDR.mat')
    sio.savemat(CDR_file_path, {'P': weight_})
    print(f'MATLAB .mat file saved at: {CDR_file_path}')


    #similar for MMSE
    ADNI_idx = np.where(np.array(ADNI_MMSE_regions) == 1)[0]
    OASIS_idx = np.where(np.array(OASIS_MMSE_regions) == 1)[0]

    common_idx = np.intersect1d(OASIS_idx, ADNI_idx)
    ADNI_union_idx = np.setdiff1d(ADNI_idx, OASIS_idx)
    OASIS_union_idx = np.setdiff1d(OASIS_idx, ADNI_idx)

    weight = np.ones([202, 1]) * -999
    weight[common_idx] = 3
    weight[ADNI_union_idx] = 2
    weight[OASIS_union_idx] = 1

    weight_ = list(weight.T[0])
    weight_ = [-999, -999] + weight_

    MMSE_file_path = os.path.join(out_dir, f'common_unique_ADNI_OASIS_MMSE.mat')
    sio.savemat(MMSE_file_path, {'P': weight_})
    print(f'MATLAB .mat file saved at: {MMSE_file_path}')
    common_unique_ADNI_OASIS_MMSE = weight_

    return common_unique_ADNI_OASIS_CDR, common_unique_ADNI_OASIS_MMSE