from model_type import ModelType
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from box import Box
from typing import Union

class ModelFactory:
    def __init__(self, model_name:str, cfg:Box):
        self.model_name: str = model_name
        self.cfg: Box = cfg

    def create_model(self) -> Union[LinearRegression, GradientBoostingRegressor, DecisionTreeRegressor, 
                                    RandomForestRegressor, MLPRegressor, xgb.XGBRegressor, lgb.LGBMRegressor]:
        if self.model_name == ModelType.LINEAR_REGRESSION.value:
            return LinearRegression()
        elif self.model_name == ModelType.GRADIENT_BOOSTING.value:
            return GradientBoostingRegressor(n_estimators=self.cfg.model.model_params.n_estimators, 
                                             random_state=self.cfg.model.model_params.random_state)
        elif self.model_name == ModelType.DECISION_TREE.value:
            return DecisionTreeRegressor(random_state=self.cfg.model.model_params.random_state)
        elif self.model_name == ModelType.RANDOM_FOREST.value:
            return RandomForestRegressor(n_estimators=self.cfg.model.model_params.n_estimators, 
                                         random_state=self.cfg.model.model_params.random_state)
        elif self.model_name == ModelType.MLP_REGRESSOR.value:
            return MLPRegressor(hidden_layer_sizes=self.cfg.model.model_params.hidden_layer_sizes, 
                                activation=self.cfg.model.model_params.activation, 
                                random_state=self.cfg.model.model_params.random_state, 
                                max_iter=self.cfg.model.model_params.max_iter)
        elif self.model_name == ModelType.XGBOOST.value:
            return xgb.XGBRegressor(n_estimators=self.cfg.model.model_params.n_estimators, 
                                    learning_rate=self.cfg.model.model_params.learning_rate, 
                                    random_state=self.cfg.model.model_params.random_state)
        elif self.model_name == ModelType.LIGHTGBM.value:
            return lgb.LGBMRegressor(n_estimators=self.cfg.model.model_params.n_estimators, 
                                     learning_rate=self.cfg.model.model_params.learning_rate, 
                                     random_state=self.cfg.model.model_params.random_state)
        else:
            raise ValueError('Model is not defined. Please check the model type.')
                                                                    
    

    
