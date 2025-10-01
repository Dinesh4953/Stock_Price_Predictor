import yaml
from stock_info.Exception.exception import StockPredictionException
from stock_info.logging.logger import logging
import os, sys
import numpy as np
import numpy as np
import dill
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, r2_score, precision_score

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