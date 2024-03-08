# test_preprocessing.py
import pandas as pd
from box import Box

from src.data_processing import one_hot_encoding, standard_scaler, Processesor

cfg_mock = Box(
    {
        "data": {
            "categorical_features": {"one_hot_encoding": ["weathersit"]},
            "numerical_features": {
                "standard_scaler": ["temp"],
                "sine_cosine_transformation": ["mnth"],
            },
            "pca_transformation": {"n_components": 2},
            "target_column": "cnt",
        }
    }
)


data_mock = pd.DataFrame(
    {
        "weathersit": [1, 2, 3, 4],
        "temp": [1, 2, 3, 4],
        "mnth": [1, 2, 3, 4],
        "cnt": [5, 6, 7, 8],
    }
)


def test_one_hot_encoding():
    result = one_hot_encoding(cfg_mock, data_mock)
    print(data_mock.columns)
    print(result.columns)
    assert "weathersit_1" in result.columns and "weathersit" not in result.columns


def test_standard_scaler():
    result = standard_scaler(cfg_mock, data_mock)
    scaled_mean = result["temp"].mean()
    assert scaled_mean < 1e-6


def test_processor_integration():
    processor = Processesor(cfg_mock)
    X, y = processor.split_feature_and_target(data_mock)
    assert (
        cfg_mock.data.target_column not in X.columns
        and y.name == cfg_mock.data.target_column
    )


def test_full_processing():
    processor = Processesor(cfg_mock)
    processed_data = processor.processing(data_mock)
    assert "PC1" in processed_data.columns
