from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from box import Box
import pandas as pd
from typing import Tuple


def one_hot_encoding(cfg: Box, data: pd.DataFrame) -> pd.DataFrame:
    one_hot_encoding_column_list = []
    for column in cfg.data.categorical_features.one_hot_encoding:
        one_hot_encoding_column_list.append(column)

    if len(one_hot_encoding_column_list) == 0:
        return data

    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(data[one_hot_encoding_column_list]).toarray()
    columns = encoder.get_feature_names_out(input_features=one_hot_encoding_column_list)
    df_encoded = pd.DataFrame(encoded_data, columns=columns)

    data_dropped = data.drop(one_hot_encoding_column_list, axis=1)
    data_final = pd.concat([data_dropped, df_encoded], axis=1)

    return data_final


def standard_scaler(cfg: Box, data: pd.DataFrame) -> pd.DataFrame:
    standard_scaler_column_list = []
    for column in cfg.data.numerical_features.standard_scaler:
        standard_scaler_column_list.append(column)

    if len(standard_scaler_column_list) == 0:
        return data

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[standard_scaler_column_list])
    df_scaled = pd.DataFrame(scaled_data, columns=standard_scaler_column_list)

    data_dropped = data.drop(standard_scaler_column_list, axis=1)
    data_final = pd.concat([data_dropped, df_scaled], axis=1)

    return data_final


def sine_cosine_transformation(cfg: Box, data: pd.DataFrame) -> pd.DataFrame:
    sine_cosine_transformation_column_list = []
    for column in cfg.data.numerical_features.sine_cosine_transformation:
        sine_cosine_transformation_column_list.append(column)

    if len(sine_cosine_transformation_column_list) == 0:
        return data

    for column in sine_cosine_transformation_column_list:
        data[column + "_sin"] = np.sin((2 * np.pi * data[column]) / max(data[column]))
        data[column + "_cos"] = np.cos((2 * np.pi * data[column]) / max(data[column]))

    data_dropped = data.drop(sine_cosine_transformation_column_list, axis=1)

    return data_dropped


def pca_transformation(cfg: Box, data: pd.DataFrame) -> pd.DataFrame:
    if cfg.data.pca_transformation.n_components == 0:
        return data
    pca = PCA(n_components=cfg.data.pca_transformation.n_components)
    X_pca = pca.fit_transform(data)
    df_pca = pd.DataFrame(
        data=X_pca,
        columns=[f"PC{i+1}" for i in range(cfg.data.pca_transformation.n_components)],
    )

    return df_pca


class Processesor:
    def __init__(self, cfg: Box):
        self.cfg: Box = cfg

    def processing(self, data: pd.DataFrame) -> pd.DataFrame:
        data = one_hot_encoding(self.cfg, data)
        data = standard_scaler(self.cfg, data)
        data = sine_cosine_transformation(self.cfg, data)
        data = pca_transformation(self.cfg, data)
        return data

    def split_feature_and_target(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        X = data.drop(self.cfg.data.target_column, axis=1)
        y = data[self.cfg.data.target_column]
        return X, y
