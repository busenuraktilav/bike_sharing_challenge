from src.metric_type import MetricType
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator
import numpy as np


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


class Trainer:
    def __init__(self, model: BaseEstimator):
        self.model: BaseEstimator = model

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> BaseEstimator:
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(
        self, y_test: np.ndarray, preds: np.ndarray, metric_name: str
    ) -> float:
        if metric_name == MetricType.MSE.value:
            return mean_squared_error(y_test, preds)
        elif metric_name == MetricType.MAE.value:
            return mean_absolute_error(y_test, preds)
        elif metric_name == MetricType.RMSE.value:
            return root_mean_squared_error(y_test, preds)
        else:
            raise ValueError("Metric is not defined. Please check the metric type.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
