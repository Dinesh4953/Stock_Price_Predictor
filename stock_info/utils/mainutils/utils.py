import yaml
from stock_info.Exception.exception import StockPredictionException
from stock_info.logging.logger import logging
import os, sys
import numpy as np
import numpy as np
import dill
import pickle

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score 

def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise StockPredictionException(e, sys) 

def write_yaml_file(file_path:str, content :object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open (file_path, "w") as file:
            yaml.dump(content, file)
            
    except Exception as e:
        raise StockPredictionException(e, sys)
    
    
def save_numpy_array_data(file_path:str, array:np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise StockPredictionException(e, sys)
    
    
## pkl object saving
def save_object(file_path:str, obj:object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils Class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj=obj, file=file_obj)
        logging.info("Exited the save_object method of MainUtils Class")
    except Exception as e:
        raise StockPredictionException(e, sys)
    
## To read pickel file
def load_object(file_path:str,) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise StockPredictionException(e, sys)
                

def load_numpy_array_data(filepath:str) -> np.array :
    try:
        with open(filepath, "rb") as file:
            return np.load(file)
    except Exception as e:
        raise StockPredictionException(e, sys) 
                

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            
            gs = RandomizedSearchCV(model, para, cv=3,n_jobs=-1, random_state=42)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            report[list(models.keys())[i]] =  test_r2
            
        return report
    except Exception as e:
        raise StockPredictionException(e , sys)

                