from stock_info.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

import os
import sys

from stock_info.Exception.exception import StockPredictionException
from stock_info.logging.logger import logging

class StockModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise StockPredictionException(e, sys)
        
    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise StockPredictionException(e, sys)