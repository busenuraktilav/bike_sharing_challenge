from enum import Enum

class ModelType(Enum):
    LINEAR_REGRESSION = 'linear_regression'
    GRADIENT_BOOSTING = 'gradient_boosting'
    DECISION_TREE = 'decision_tree'
    RANDOM_FOREST = 'random_forest'
    MLP_REGRESSOR = 'mlp_regressor'
    XGBOOST = 'xgboost'
    LIGHTGBM = 'lightgbm'
