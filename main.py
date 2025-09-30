from stock_info.components.data_ingestion import DataIngestion, DataIngestionArtifact
from stock_info.Exception.exception import StockPredictionException
from stock_info.logging.logger import logging
from stock_info.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
import sys
from datetime import datetime

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig(datetime.now())
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info("Initiated the data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info("Data Ingestion is Completed")
    except Exception as e:
        raise StockPredictionException(e, sys)
        
