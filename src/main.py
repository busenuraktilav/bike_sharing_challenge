from ucimlrepo import fetch_ucirepo 
from box import Box
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from data_processing import Processesor
from model import ModelFactory
from engine import Trainer

def fetch_data(id:int=275) -> pd.DataFrame:
    bike_sharing_dataset = fetch_ucirepo(id=id)
    X = bike_sharing_dataset.data.features 
    y = bike_sharing_dataset.data.targets 
    df = pd.DataFrame(X, columns=['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed'])
    df['cnt'] = y
    return df

def load_config(config_file:str) -> Box:
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
        cfg = Box(config_data)
    return cfg

def main() -> None:
    cfg = load_config('./cfg.yaml')
    df = fetch_data()

    processor = Processesor(cfg)
    X, y = processor.split_feature_and_target(df)
    X = processor.processing(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=cfg.training.test_size, 
                                                        random_state=cfg.model.model_params.random_state)

    model = ModelFactory(cfg.model.model_type, cfg).create_model()

    trainer = Trainer(model)
    trainer.train(X_train, y_train)
    mae = trainer.evaluate(y_test, trainer.predict(X_test), cfg.evaluation.metric)
    print(f'Mean Absolute Error: {mae}')




if __name__ == "__main__":

    main()