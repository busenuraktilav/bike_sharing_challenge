from enum import Enum

class ModelType(Enum):
    LINEAR_REGRESSION = 'linear_regression'
    DECISION_TREE = 'decision_tree'
    RANDOM_FOREST = 'random_forest'
    MLP_REGRESSOR = 'mlp_regressor'
    XGBOOST = 'xgboost'
    LIGHTGBM = 'lightgbm'
