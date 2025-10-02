from stock_info.components.data_ingestion import DataIngestion, DataIngestionArtifact
from stock_info.components.data_validation import DataValidation, DataValidationArtifact
from stock_info.components.data_transformation import DataTransformation, DataTransformationArtifact
from stock_info.Exception.exception import StockPredictionException
from stock_info.logging.logger import logging
from stock_info.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig
import sys
from datetime import datetime

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig(datetime.now())
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info("Initiated the data ingestion")
        data_ingestion_artifact:DataIngestionArtifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        print("--------------------------------------------------------------------------")
        logging.info("Data Ingestion is Completed")
        
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
        logging.info("Initiated data validation")
        data_validation_artifact:DataValidationArtifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)
        print("--------------------------------------------------------------------------")
        logging.info("Data Validation is Completed")
        
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
        logging.info("Data Transformation Initiated ")
        data_transformation_artifact:DataTransformationArtifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Data Transformation completed")
        print("--------------------------------------------------------------------------")

        
    except Exception as e:
        raise StockPredictionException(e, sys)
        
