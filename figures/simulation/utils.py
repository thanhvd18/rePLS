import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figure3 as fig3
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline,make_pipeline
from rePLS import rePLS,rePCR,reMLR
from scipy.stats import t,pearsonr
import random


def generate_data(N:int, I:int, J:int,SNR_ZX:float, Z_affect_regions:float,
                    SNR_ZY, SNR_XY, random_seed=1):
    np.random.seed(random_seed)
    params = {
        'gender_ratio': 0.3,
        'age_group_weight': np.array([10, 20, 70]) / 100,  # 40-60-80 age group percentage
        'normalized': False
    }

    Z = gen_confounder(N, **params)
    # Z = Z / np.linalg.norm(Z, axis=0)

    # Simulate X
    params = {
        'SNR_ZX': SNR_ZX,
        'seed': random_seed,
        'Z_affect_regions': Z_affect_regions,
        #     'principle_component':5
    }
    X, X_true, affected_brain_map = gen_X(I, Z, **params)

    params = {
        'SNR_ZY': SNR_ZY,  # /f_ZY
        'SNR_XY': SNR_XY  # Y_noise
    }
    PQ_true = np.random.rand(I, J)
    Y, Y_true, Y_noise = gen_Y(X_true, Z, PQ_true, **params)

    return X, X_true, Y, Y_true, Z

def gen_Y(X_true, Z, PQ_true, **params):
    """
        Input:
            J: number of outcomes
            Z: confounder matrix
            X_true: input matrix that directly effect Y

            params:


        Return: Y: JxN matrix

    """

    N, _ = X_true.shape
    _, R = Z.shape
    I, J = PQ_true.shape

    # the output matrix that is not effected by the confounder matrix
    Y_true = X_true @ PQ_true

    Y_noise = np.random.randn(*Y_true.shape)
    Y_noise = Y_noise * (np.linalg.norm(Y_true, axis=0) /
                         (np.linalg.norm(Y_noise, axis=0) * 10**(params['SNR_XY']/10)))

    # check SNR fomular
    assert abs(params['SNR_XY'] - 10 *
               np.log10(np.linalg.norm(Y_true)/np.linalg.norm(Y_noise))) < 1e-6

    # relationship between Z and Y: linear mapping
    ZY = np.random.randn(R, J)
    f_ZY = Z@ZY
    f_ZY = f_ZY * (np.linalg.norm(Y_true, axis=0) /
                   (np.linalg.norm(f_ZY, axis=0) * 10**(params['SNR_ZY']/10)))

    assert abs(params['SNR_ZY'] - 10 *
               np.log10(np.linalg.norm(Y_true)/np.linalg.norm(f_ZY))) < 1e-6
    Y = Y_true + f_ZY + Y_noise

    return Y, Y_true, Y_noise



def gen_X(I, Z, **params):
    N, R = Z.shape

    X_true = np.random.randn(N, I)
    #     X_true[:,:2] = 0

    # relationship between Z and X: linear mapping
    # impose effect of Z on P, PQ
    np.random.seed(params['seed'])
    number_affected_region = int(params['Z_affect_regions'] * I)
    affected_regions = np.random.randint(0, I, number_affected_region)
    ZX = np.random.randn(R, number_affected_region)
    f_ZX = Z @ ZX
    f_ZX = f_ZX * (np.linalg.norm(X_true[:, affected_regions], axis=0) / (
                np.linalg.norm(f_ZX, axis=0) * 10 ** (params['SNR_ZX'] / 10)))

    # check SNR fomular
    assert abs(params['SNR_ZX'] - 10 * np.log10(np.linalg.norm(X_true[:, affected_regions]) / np.linalg.norm(f_ZX))) < 1e-6
    X = X_true.copy();
    X[:, affected_regions] = X_true[:, affected_regions] + f_ZX

    affected_brain_map = np.zeros(I)
    affected_brain_map[affected_regions] = 1 * np.sign((sum(X_true[:, affected_regions])))
    return X, X_true, affected_brain_map


def gen_P_PQ(X_true, J, **params):
    from sklearn.decomposition import SparsePCA
    N, I = X_true.shape
    K = params['principle_component']
    # generate P_true, PQ_true in PLS
    #     sparse_pca = SparsePCA(n_components=K,alpha=0.95)
    #     sparse_pca.fit(X_true)

    #     P_true = sparse_pca.components_
    from sklearn.decomposition import PCA
    pca = PCA(n_components=K)
    pca.fit(X_true)
    P_true = pca.components_
    alpha = np.eye(K)
    Q = np.random.rand(J, K)
    PQ_true = P_true.T @ alpha @ Q.T
    return P_true, PQ_true


def gen_confounder(N, **params):
    '''
        Input:
            N: number of samples
            R: number of confounders
            params (dict):
                'gender_ratio': [0-1]
                'age_group_weight':

        Output: confounder matrix Z: NxR matrix
    '''
    import random
    gender_ratio = params['gender_ratio']
    age_group_weight = params['age_group_weight']
    age_group_sample = [int(x*N) for x in age_group_weight]
    Z = np.zeros((N, 2))

    Z0 = [0]*int(N*gender_ratio) + [1]*(N-int(N*gender_ratio))
    random.shuffle(Z0)
    Z[:, 0] = Z0

    Z1 = np.random.randint(30, 50, size=(age_group_sample[0],)).tolist() + np.random.randint(51, 70, size=(
        age_group_sample[1],)).tolist() + np.random.randint(71, 90, size=(age_group_sample[2],)).tolist()
    random.shuffle(Z1)
    Z[:, 1] = Z1
    # if params['normalized']:
    # Z = Z/np.linalg.norm(Z,axis=0)
    return Z

def boostrapping_prediction(X,Y,Y_true,Z,method,N_repeats):
    df_repeated = pd.DataFrame(columns=["r", "MSE", "p_value"])
    for random_state in range(N_repeats):
        params = {
            'population_rate': 0.8,
            'random_state': random_state
        }
        n_components = 5
        X_train, X_test, Y_train, Y_test, Z_train, Z_test = boostrapping(X, Y, Y_true, Z, **params)

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

        result_j = fig3.utils.cal_correlation_MSE_regression(Y_test, y_pred)
        df_i = pd.DataFrame(result_j).T
        df_i.columns = ['r', 'MSE', 'p_value']
        if random_state == 0:
            df_repeated = df_i
        else:
            df_repeated += df_i
    df_repeated /= N_repeats
    df_repeated = df_repeated.mean()


    return df_repeated


def boostrapping(X, Y, Y_true, Z, **params):
    '''
        Inputs:
            X: input matrix
            Y: output matrix
            Y_true
            Z:
        Output:
             X_train,X_test,Y_train, Y_test,Z_train, Z_test
    '''
    N = X.shape[0]
    sample_size = int(params['population_rate']*N)
    indicies = list(range(N))
    boot_indicies = resample(
        indicies, replace=True, n_samples=sample_size, random_state=params['random_state'])
    oob_indicies = [x for x in indicies if x not in boot_indicies]
    X_train = X[boot_indicies]
    X_test = X[oob_indicies]
    Y_train = Y[boot_indicies]
    Y_test = Y_true[oob_indicies]

    Z_train = Z[boot_indicies]
    Z_test = Z[oob_indicies]

    return X_train, X_test, Y_train, Y_test, Z_train, Z_test





def gen_P_PQ(X_true, J, **params):
    from sklearn.decomposition import SparsePCA
    N, I = X_true.shape
    K = params['principle_component']
    # generate P_true, PQ_true in PLS
    sparse_pca = SparsePCA(n_components=K, alpha=0.95)
    sparse_pca.fit(X_true)

    P_true = sparse_pca.components_
    alpha = np.eye(K)
    Q = np.random.rand(J, K)
    PQ_true = P_true.T@alpha@Q.T
#     PQ_true = min_max_normalize(PQ_true)
    return P_true, PQ_true


def gen_Y(X_true,Z,PQ_true,**params):
    """
        Input:
            J: number of outcomes
            Z: confounder matrix
            X_true: input matrix that directly effect Y

            params:


        Return: Y: JxN matrix

    """

    N,_ = X_true.shape
    _,R = Z.shape
    I,J = PQ_true.shape


    # the output matrix that is not effected by the confounder matrix
    Y_true = X_true @ PQ_true

    Y_noise = np.random.randn(*Y_true.shape)
    Y_noise = Y_noise* (np.linalg.norm(Y_true, axis=0)/ (np.linalg.norm(Y_noise, axis=0)* 10**(params['SNR_XY']/10)))

    #check SNR fomular
    assert  abs(params['SNR_XY'] - 10*np.log10(np.linalg.norm(Y_true)/np.linalg.norm(Y_noise))) <1e-6



    #relationship between Z and Y: linear mapping
    ZY = np.random.randn(R,J)
    f_ZY =  Z@ZY
    f_ZY =  f_ZY * (np.linalg.norm(Y_true, axis=0)/ (np.linalg.norm(f_ZY, axis=0)* 10**(params['SNR_ZY']/10)))

    assert  abs(params['SNR_ZY'] - 10*np.log10(np.linalg.norm(Y_true)/np.linalg.norm(f_ZY))) <1e-6
    Y = Y_true + f_ZY + Y_noise
    Y_true = Y_true + f_ZY

    return Y,Y_true,Y_noise