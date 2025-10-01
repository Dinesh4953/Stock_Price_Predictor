from stock_info.entity.artifact_entity import  DataValidationArtifact, DataIngestionArtifact
from stock_info.entity.config_entity import DataValidationConfig

from stock_info.Exception.exception import StockPredictionException
from stock_info.logging.logger import logging

from scipy.stats import ks_2samp

from stock_info.constants.training_pipeline import SCHEMA_FILE_PATH

from stock_info.utils.mainutils.utils import read_yaml_file, write_yaml_file

import pandas as pd
import os, sys


class DataValidation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            
        except Exception as e:
            raise StockPredictionException(e, sys)
    
    @staticmethod
    def read_data(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise StockPredictionException(e, sys)
        
    def validate_number_of_columns(self, dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self.schema_config['columns'])
            logging.info(f"Required numnber of columns: {number_of_columns}")
            logging.info(f"Data frame has columns : {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception  as e:
            raise StockPredictionException(e, sys)

    def detect_dataset_drift(self, base_df,  current_df, threshold=0.05)-> bool:
        #In machine learning and data pipelines, we often assume that the data we train on (base_df) and 
        #the data we apply the model to (current_df) come from the same statistical distribution. 
        #If that assumption breaks, the model might make poor predictions. This is called dataset drift or data distribution shift.

        try:
            status = True
            report = {}
            for col in base_df.columns:
                d1 = base_df[col]
                d2 = current_df[col]
                is_sample_dist = ks_2samp(d1, d2)
                if threshold<=is_sample_dist.pvalue:
                    isfound = False
                else:
                    isfound = True
                    status = False
                report.update(
                    {
                        col:{
                            "p_value":float(is_sample_dist.pvalue),
                            "drift_status": isfound
                        }
                    }
                )
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            
            ## Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            
        except Exception as e:
            raise StockPredictionException(e, sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            ## read data from train and test
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            
            
            ## Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                logging.error("Train dataframe does not contain all columns.")
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                logging.error(" Test dataframe does not contain all columns.")
                
            status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )
            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
        except Exception as e:
            raise StockPredictionException(e, sys)