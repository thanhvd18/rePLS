from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from icecream import ic
import os


features = ['Schaefer_200_7']
outcomes = [
'CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE',
'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_perc_forgetting'
]
confounders = ["PTGENDER", "AGE"]
timepoints = ['bl', 'm06', 'm12', 'm18', 'm24',
          'm36', 'm48', 'm60', 'm72', 'm84', 'm96', 'm108']
# Filter and preprocess the dataset
base_fields = ["SubjectID", "ScanDate", "UID", "VISCODE",
           "image_path", 'DX', 'DX_bl', features[0]]
filtered_fields = base_fields + outcomes + confounders


def plot_boxplot_8outcomes(Y: np.ndarray,Z: np.ndarray) -> plt.Axes:
    """
    Plots a boxplot representing 8 outcomes with specific transformations and filters applied to the dataset.

    Combines two input dataframes, applies mapping and transformations on categorical columns, reshapes data
    for plotting, and generates a boxplot using the seaborn library. Specific outcomes are filtered for
    representation in the plot, and formatting is applied for style and visual clarity.

    Arguments:
        Y (np.ndarray): Input dataframe or ndarray containing the first set of variables and features for transformation.
        Z (np.ndarray): Input dataframe or ndarray containing the second set of variables and features for concatenation
                        with the first.

    Returns:
        plt.Axes: Matplotlib Axes object of the generated boxplot.
    """
    df_YZ = pd.concat([Y,Z], axis=1)
    df_YZ['gender'] = df_YZ.PTGENDER.map({df_YZ.PTGENDER.unique()[i]: ["Female", "Male"][i] for i in range(2)})
    df_YZ["age_group"] = df_YZ.AGE.apply(lambda x: 0 if (x >= 50 and x <= 60) else
    (1 if (x > 60 and x <= 70)
     else (2 if (x > 70 and x <= 80)
           else 3)))

    df_YZ['idx'] = df_YZ.index
    df_all_longform = df_YZ.melt(value_vars=outcomes, id_vars='idx')
    df_all_longform = df_all_longform[(df_all_longform.variable != 'RAVLT_perc_forgetting') | (
            (df_all_longform.variable == 'RAVLT_perc_forgetting') & (df_all_longform.value > -75))]
    sns.set(font_scale=3.5)
    fig = plt.figure(figsize=(30, 20))
    ax = sns.boxplot(x="variable", y="value", hue="variable", data=df_all_longform, showfliers=False,
                      palette=['#03a9e4ff' for i in range(len(outcomes))], legend=False)
    ax.grid(False)
    # ic(df_all_longform.shape,df_YZ.shape) # df_all_longform.shape: (22891, 3), df_YZ.shape: (2862, 13)
    return fig, ax

def plot_boxplot_7network(network_df: pd.DataFrame) -> Tuple[plt.Axes, pd.DataFrame]:
    """
    Plot a boxplot for the given network data.

    This function generates a boxplot for the input network dataframe using
    a predefined network order and corresponding color palette. It uses seaborn
    for the visualization and assumes a specific structure and ordering of the
    network names.

    Parameters:
        network_df (pd.DataFrame): A pandas DataFrame containing the data for
        the seven networks. The columns should correspond to the predefined
        network names in the required order.

    Returns:
        plt.Axes: The matplotlib Axes object for the generated boxplot.
    """
    sns.set(font_scale=1.5)
    fig = plt.figure(figsize=(15, 5))
    colors = ['#650871', '#97d7d7', '#235731', '#d30589', '#555533', '#e38421', '#c53a48']
    network_name7_order = ['SalVentAttn', 'Limbic', 'SomMot', 'DorsAttn', 'Vis', 'Cont', 'Default']
    network_name7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    network_name7_idx = [network_name7.index(x) for x in network_name7_order]
    colors = [colors[x] for x in network_name7_idx]
    network_name7_order = ['SalVentAttn', 'Limbic', 'SomMot', 'DorsAttn', 'Vis', 'Cont', 'Default']
    ax = sns.boxplot(data=network_df[network_name7_order], palette=colors)
    ax.grid(False)
    return fig, ax

# def get_7_network(X: pd.DataFrame,lut_path:str) -> pd.DataFrame:
#     """
#         Maps input data to the 7-network brain parcellation framework using a lookup
#         table. Aggregates the data for each network by calculating the mean of the
#         features corresponding to that network.
#
#         Parameters
#         ----------
#         X : np.ndarray
#             Input data matrix where rows represent samples and columns represent
#             features.
#         lut_path : str
#             File path to the lookup table describing the mapping of features to
#             their respective networks.
#
#         Returns
#         -------
#         pd.DataFrame
#             A DataFrame where each column represents one of the 7 brain networks
#             and contains the aggregated mean values for the given network computed
#             across the features. Rows correspond to samples.
#
#         Raises
#         ------
#         ValueError
#             If the input lookup table is not in the expected format or if the
#             required network mapping cannot be created due to missing data.
#         IOError
#             If the provided lookup table file path is invalid or if there are issues
#             reading the file.
#
#         Notes
#         -----
#         The 7-network parcellation framework divides the brain into seven functional
#         networks: Visual (Vis), Somato-Motor (SomMot), Dorsal Attention (DorsAttn),
#         Salience/Ventral Attention (SalVentAttn), Limbic, Control (Cont), and Default
#         Mode (Default). Mapping is performed by averaging feature values in the input
#         data corresponding to a network as determined by the lookup table.
#     """
#     network_name7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
#     network_name17 = ["VisCent", "VisPeri", "SomMotA", "SomMotB", "DorsAttnA", "DorsAttnB", "SalVentAttnA",
#                       "SalVentAttnB", "LimbicA", "LimbicB", "ContA", "ContB", "ContC", "DefaultA", "DefaultB",
#                       "DefaultC", "TempPar"]
#
#     parcellation_df =  get_7_network_df(lut_path)
#     seven_network_df = pd.DataFrame({})
#     network = 7
#     for i in range(network):
#         col_network = list(parcellation_df[parcellation_df.network == i].index)
#         col_network = [x + 1 for x in col_network]
#
#     return col_network

def mapping_to_7_network(X: pd.DataFrame,seven_network_path:str) -> pd.DataFrame:
    """
        Maps input data to the 7-network brain parcellation framework using a lookup
        table. Aggregates the data for each network by calculating the mean of the
        features corresponding to that network.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix where rows represent samples and columns represent
            features.
        seven_network_path : str
            File path to the lookup table describing the mapping of index of region to
            their respective networks.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each column represents one of the 7 brain networks
            and contains the aggregated mean values for the given network computed
            across the features. Rows correspond to samples.

        Raises
        ------
        ValueError
            If the input lookup table is not in the expected format or if the
            required network mapping cannot be created due to missing data.
        IOError
            If the provided lookup table file path is invalid or if there are issues
            reading the file.

        Notes
        -----
        The 7-network parcellation framework divides the brain into seven functional
        networks: Visual (Vis), Somato-Motor (SomMot), Dorsal Attention (DorsAttn),
        Salience/Ventral Attention (SalVentAttn), Limbic, Control (Cont), and Default
        Mode (Default). Mapping is performed by averaging feature values in the input
        data corresponding to a network as determined by the lookup table.
    """
    network_name7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    network_name17 = ["VisCent", "VisPeri", "SomMotA", "SomMotB", "DorsAttnA", "DorsAttnB", "SalVentAttnA",
                      "SalVentAttnB", "LimbicA", "LimbicB", "ContA", "ContB", "ContC", "DefaultA", "DefaultB",
                      "DefaultC", "TempPar"]

    # parcellation_df =  get_7_network_df(seven_network_path)
    parcellation_df = pd.read_csv(seven_network_path)
    seven_network_df = pd.DataFrame({})
    network = 7
    for i in range(network):
        # col_network = list(parcellation_df[parcellation_df.network == i]["index"].values)
        col_network = list(parcellation_df[parcellation_df.network == i].index.values)
        col_network = [x + 1 for x in col_network]
        seven_network_df[network_name7[i]] = X.iloc[:, col_network].mean(axis=1).tolist() #axis=1: Calculate the mean across the columns (i.e., for each row).
    return seven_network_df

def get_7_network_df(lut_path):
    """
    Processes a lookup table (LUT) file to associate network information with each parcellation.

    Parameters:
        lut_path (str): Path to the LUT file containing parcellation information with columns
            'index', 'x', 'y', 'z', and 'name', separated by spaces.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the parsed parcellation data, including an
            additional 'network' column derived by mapping the 'name' field to the 7-network
            scheme.
    """
    network_name7 = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    parcellation_df  = pd.read_csv(lut_path,sep = ' ', names = ['index', 'x', 'y', 'z', 'name'])
    parcellation_df["network"] = parcellation_df.name.apply(lambda x: name2network(name=x, network_name=network_name7))
    return parcellation_df


def name2network(name: str, network_name: List[str]) -> Optional[int]:
    """
    Finds the index of the first network name in the list that appears in the input name.

    Parameters:
    - name (str): The name to search within.
    - network_name (List[str]): A list of network names to search for.

    Returns:
    - Optional[int]: The index of the first matching network name, or None if no match is found.
    """
    for i, n in enumerate(network_name):
        if n in name:
            return i
    return None

def get_input_output_confounder(df: pd.DataFrame) -> List[np.ndarray]:
    X = [eval(x) for x in df.loc[:, features[0]]]
    X = pd.DataFrame(X).astype(float)
    Y = df[outcomes].astype(float)
    Z = df[confounders].astype(float)
    return X,Y,Z

def categorize_disease_group(df: pd.DataFrame) ->Tuple[pd.DataFrame, List[str], List[int]] :
    """
    Categorizes disease groups for subjects based on clinical diagnosis data within a DataFrame.

    This function processes a DataFrame with subject diagnosis information to categorize subjects into
    specific disease stages ("CN", "AD", "MCI", "pMCI", and "sMCI"). It creates groupings based on the
    current and baseline diagnosis, and adjusts subject distribution accordingly to ensure proper
    categorization. The function then generates labels for each group, combines the labels with the
    original DataFrame for further analysis, and returns the modified DataFrame along with selected
    subject IDs and group labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns 'SubjectID', 'DX', 'DX_bl', and potentially others. These
        columns represent the subject identification, current diagnosis, and baseline diagnosis.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[int]]
        A tuple containing the following:
        - Updated DataFrame including disease group labels for subjects.
        - List of unique subject IDs categorized in the disease groups.
        - List of labels corresponding to the disease groups for the categorized subjects.
    """
    # Define subject groups
    df1 = df[['SubjectID', 'DX', 'DX_bl']]
    df['DX_encode'] = df['DX'].map({'CN': 0, 'MCI': 1, 'AD': 2})
    df['DX_one_hot'] = df['DX'].map({'CN': [1,0,0], 'MCI': [0,1,0], 'AD': [0,0,1]})

    subject_groups = {
        'CN': set(df1[(df1['DX'] == 'CN') & (df1['DX_bl'] == 'CN')]['SubjectID']),
        'AD': set(df1[(df1['DX'] == 'AD') & (df1['DX_bl'] == 'AD')]['SubjectID']),
        'MCI': set(df1[(df1['DX'] == 'MCI') & (df1['DX_bl'].str.contains('MCI'))]['SubjectID']),
        'pMCI': set(df1[(df1['DX'] == 'AD') & (df1['DX_bl'].str.contains('MCI'))]['SubjectID'])
    }
    subject_groups['sMCI'] = subject_groups['MCI'] - subject_groups['pMCI']
    df_pMCI = df[df['SubjectID'].isin(subject_groups['pMCI'])]
    subject_groups['CN'] -= set(df_pMCI[df_pMCI['DX'].isin(['MCI', 'AD'])]
                                ['SubjectID'])
    subject_groups['sMCI'] -= set(df_pMCI[df_pMCI['DX'] == 'CN']['SubjectID'])
    # Prepare labels for subject groups
    df_stage = pd.DataFrame(columns=['subject', 'stage'])
    for i, (stage, subjects) in enumerate(subject_groups.items()):
        for subject in subjects:
            label = [0] * len(subject_groups)
            label[i] = 1
            df_stage.loc[len(df_stage)] = [subject, label]
    df = df.merge(df_stage, left_on='SubjectID',
                  right_on='subject').drop('subject', axis=1)
    df.drop_duplicates(subset=['UID'], inplace=True)
    subjects = [subject for group in subject_groups.values()
                for subject in group]
    selected_subjects = list(subject_groups['CN']) + list(subject_groups['sMCI']) + \
                        list(subject_groups['pMCI']) + list(subject_groups['AD'])


    labels = [0] * len(subject_groups['CN']) + \
             [1] * len(subject_groups['sMCI']) + \
             [2] * len(subject_groups['pMCI']) + \
             [3] * len(subject_groups['AD'])
    return df, selected_subjects, labels


def preprocess_df(data_path:str) -> pd.DataFrame:
    """
    Preprocesses a CSV dataset and returns a cleaned DataFrame with selected features,
    outcomes, confounders, and additional transformations.

    This function reads a CSV file from the provided file path, selects relevant columns
    (fields), filters the dataset by dropping rows with missing values or duplicates, and
    encodes categorical variables such as gender. Additionally, it categorizes the age
    field into predefined age groups.

    Parameters
    ----------
    data_path : str
        The file path to the CSV file.

    Returns
    -------
    pd.DataFrame
        A preprocessed Pandas DataFrame containing selected features, outcomes,
        confounders, and transformed variables.
    """
    df = pd.read_csv(data_path, index_col=False, low_memory=False)
    # Define feature, outcome, and confounder variables
    df = df[filtered_fields].dropna().drop_duplicates(
        subset=['SubjectID', 'ScanDate'])

    #encode sex and age
    df['PTGENDER'] = df['PTGENDER'].map({'Female': '0', 'Male': '1'})

    df["age_group"] = df['AGE'].apply(lambda x: 0 if 50 <= x <= 60 else (
        1 if 60 < x <= 70 else (2 if 70 < x <= 80 else 3)))

    return df