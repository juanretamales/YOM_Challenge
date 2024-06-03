import pandas as pd
from typing import Tuple
import numpy as np
import os

import joblib

def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset from a specified file path and return the features and target variables.

    This function reads a dataset from the provided a folder path and splits the CSV it into features (X) and target (y).
    It assumes that the dataset is in a format CSV readable by pandas and that the target variable
    is the last column.

    Search for these CSV:

    Args:
        path (str): The file path to the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the features (X) and the target variable (y).
            - X (numpy.ndarray): The features from the dataset.
            - y (numpy.ndarray): The target variable from the dataset.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the dataset cannot be properly parsed or is missing expected columns.

    Example:
        >>> X, y = load_dataset('data/dataset.csv')
        >>> print(X.shape)
        (150, 4)
        >>> print(y.shape)
        (150,)
    """

    df_todotipo=pd.read_csv('data_todotipo.csv',encoding='utf-8',sep=',')
    del df_todotipo['Unnamed: 0']
    df_todotipo=df_todotipo.dropna(how='any',axis=0).copy()
    df_reggaeton=pd.read_csv('data_reggaeton.csv',encoding='utf-8',sep=',')
    del df_reggaeton['Unnamed: 0']
    del df_todotipo['time_signature']
    df_reggaeton['reggaeton']=1
    df_todotipo['reggaeton']=0
    df_reggaeton['popularity']=df_reggaeton.popularity.astype('float64')
    df_reggaeton['duration']=df_reggaeton.duration.astype('float64')
    df=pd.concat([df_reggaeton,df_todotipo]).copy()
    df.loc[:,"key_code"] = df.key.astype('category')
    df.loc[:,"mode_code"] = df['mode'].astype('category')
    del df['key']
    del df['mode']
    del df['id_new']
    df.loc[:,"reggaeton"] = df.reggaeton.astype('category')
    df.loc[:,'loudness_scale']=MinMaxScaler().fit_transform(df[['loudness']])
    df.loc[:,'tempo_scale']=MinMaxScaler().fit_transform(df[['tempo']])
    del df['loudness']
    del df['tempo']
    X=df.loc[:, df.columns != 'reggaeton']
    y=df.reggaeton.ravel()
    del X['key_code']
    del X['duration']
    del X['popularity']
    del X['mode_code']
    del X['instrumentalness']
    del X['loudness_scale']
    del X['liveness']

def load_dataset_from_folder(folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load datasets from all CSV files in the specified folder and return the features and target variables.

    This function reads all CSV files from the provided folder path, processes the datasets by cleaning
    and combining them, and then splits them into features (X) and target (y). It assumes that the target
    variable is 'reggaeton'.

    Args:
        folder_path (str): The folder path containing the dataset CSV files.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the features (X) and the target variable (y).
            - X (numpy.ndarray): The features from the dataset.
            - y (numpy.ndarray): The target variable from the dataset.

    Raises:
        FileNotFoundError: If the folder does not exist or contains no CSV files.
        ValueError: If the datasets cannot be properly parsed or are missing expected columns.

    Example:
        >>> X, y = load_dataset_from_folder('data')
        >>> print(X.shape)
        (300, 10)
        >>> print(y.shape)
        (300,)
    """
    import pandas as pd
    import os
    from sklearn.preprocessing import MinMaxScaler

    # Verify folder existence
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The folder at path {folder_path} does not exist.")
    
    # List all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the folder {folder_path}.")

    # Read and concatenate all CSV files
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, file), encoding='utf-8', sep=',')
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']
        dataframes.append(df)
    
    # Concatenate all dataframes
    df = pd.concat(dataframes, ignore_index=True)

    # Cleaning and preprocessing
    df = df.dropna(how='any', axis=0).copy()
    if 'time_signature' in df.columns:
        del df['time_signature']
    
    if 'data_reggaeton.csv' in csv_files:
        df['reggaeton'] = df['reggaeton'].fillna(0).astype(int)
    else:
        df['reggaeton'] = 0
    
    # Asign type for transform
    df['reggaeton'] = df['reggaeton'].astype('category')
    df['popularity'] = df['popularity'].astype('float64')
    df['duration'] = df['duration'].astype('float64')
    df.loc[:, "key_code"] = df['key'].astype('category').cat.codes
    df.loc[:, "mode_code"] = df['mode'].astype('category').cat.codes

    # Initialize MinMaxScaler for loudness
    loudness_scaler_file = os.path.join(folder_path, 'loudness_min_max_scaler.save')
    if os.path.exists(scaler_file):
        loudness_scaler = joblib.load(scaler_file)
    else:
        loudness_scaler = MinMaxScaler()

    # Initialize MinMaxScaler for tempo
    tempo_scaler_file = os.path.join(folder_path, 'tempo_min_max_scaler.save')
    if os.path.exists(scaler_file):
        tempo_scaler = joblib.load(scaler_file)
    else:
        tempo_scaler = MinMaxScaler()

    df.loc[:, 'loudness_scale'] = loudness_scaler.fit_transform(df[['loudness']])
    df.loc[:, 'tempo_scale'] = tempo_scaler.fit_transform(df[['tempo']])

    # Save the MinMaxScaler if not exist
    if not os.path.exists(loudness_scaler_file):
        joblib.dump(loudness_scaler, loudness_scaler_file)

    # Save the MinMaxScaler if not exist
    if not os.path.exists(tempo_scaler_file):
        joblib.dump(tempo_scaler, tempo_scaler_file)


    # Drop unnecessary columns
    columns_to_drop = ['key', 'mode', 'id_new', 'loudness', 'tempo', 'key_code', 
                       'duration', 'popularity', 'mode_code', 'instrumentalness', 
                       'liveness']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Split into features and target
    X = df.loc[:, df.columns != 'reggaeton'].values
    y = df['reggaeton'].values.ravel()

    return X, y