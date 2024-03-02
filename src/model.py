from .model_type import ModelType
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

class ModelFactory:
    def __init__(self, model_name:str):
        self.model_name = model_name

    @staticmethod
    def create_model(self, cfg):
        if self.model_name == ModelType.LINEAR_REGRESSION.value:
            return LinearRegression()
        elif self.model_name == ModelType.GRADIENT_BOOSTING.value:
            return GradientBoostingRegressor(n_estimators=cfg.model.model_params.n_estimators, random_state=cfg.model.model_params.random_state)
        elif self.model_name == ModelType.DECISION_TREE.value:
            return DecisionTreeRegressor(random_state=cfg.model.model_params.random_state)
        elif self.model_name == ModelType.RANDOM_FOREST.value:
            return RandomForestRegressor(n_estimators=cfg.model.model_params.n_estimators, random_state=cfg.model.model_params.random_state)
        elif self.model_name == ModelType.MLP_REGRESSOR.value:
            return MLPRegressor(hidden_layer_sizes=cfg.model.model_params.hidden_layer_sizes, activation=cfg.model.model_params.activation, random_state=cfg.model.model_params.random_state, max_iter=cfg.model.model_params.max_iter)
        elif self.model_name == ModelType.XGBOOST.value:
            return xgb.XGBRegressor(n_estimators=cfg.model.model_params.n_estimators, learning_rate=cfg.model.model_params.learning_rate, random_state=cfg.model.model_params.random_state)
        elif self.model_name == ModelType.LIGHTGBM.value:
            return lgb.LGBMRegressor(n_estimators=cfg.model.model_params.n_estimators, learning_rate=cfg.model.model_params.learning_rate, random_state=cfg.model.model_params.random_state)
        else:
            raise ValueError('Model is not defined. Please check the model type.')
                                                                    
    

    
