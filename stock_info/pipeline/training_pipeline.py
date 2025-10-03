import os
import sys

from stock_info.Exception.exception import StockPredictionException
from stock_info.logging.logger import logging

from stock_info.components.data_ingestion import DataIngestion
from stock_info.components.data_validation import DataValidation
from stock_info.components.data_transformation import DataTransformation
from stock_info.components.model_trainer import ModelTrainer

from stock_info.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from stock_info.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion Completed { data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise StockPredictionException(e, sys)

    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
            logging.info("Initiated data validation")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data Validation is Completed")
            return data_validation_artifact
        except Exception as e:
            raise StockPredictionException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
            logging.info("Data Transformation Initiated ")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data Transformation completed")
            return data_transformation_artifact
        except Exception as e:
            raise StockPredictionException(e, sys)

    def start_model_trainer(self, data_transformation_artifact:DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Model Training started")
            self.model_trainer_config:ModelTrainerConfig = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config=self.model_trainer_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model training artifact created")
            return model_trainer_artifact
        except Exception as e:
            raise StockPredictionException(e, sys)
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            return model_trainer_artifact
        except Exception as e:
            raise StockPredictionException(e, sys)