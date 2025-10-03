from stock_info.Exception.exception import StockPredictionException
from stock_info.logging.logger import logging

import sys
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from stock_info.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS

from stock_info.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from stock_info.entity.config_entity import DataTransformationConfig
from stock_info.utils.mainutils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise StockPredictionException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise StockPredictionException(e, sys)
        
    def get_data_transformer_object(cls) -> Pipeline:
        logging.info("Entered get_data_transformer_object method of Transformation Class")
        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(
                f"Initialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
            scaler: StandardScaler = StandardScaler()
            logging.info("Initialised StandardScaler")
            processor:Pipeline = Pipeline([("imputer",imputer),
                                         ("scaler",scaler)  ])
            return processor
        except Exception as e:
            raise StockPredictionException(e, sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiated data transformation")
        try:
            logging.info("Started data transformation ")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            train_df.drop(columns=["Date"], inplace=True)
            test_df.drop(columns=["Date"], inplace=True)
            
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            preprocessor= self.get_data_transformer_object()
            preprocessor_obj = preprocessor.fit(input_feature_train_df)
            transformed_input_train_features = preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test_features = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[transformed_input_train_features, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_features, np.array(target_feature_test_df)]
            
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_obj)
            
            save_object("final_model/preprocessor.pkl", preprocessor_obj)
            ## Preparing Artifacts
            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path= self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            
            return data_transformation_artifact
        except Exception as e:
            raise StockPredictionException(e, sys)


