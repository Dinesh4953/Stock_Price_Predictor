import os
import sys
import numpy as np
import pandas as pd

TARGET_COLUMN = "Close"
PIPELINE_NAME = "StockPredictor"
ARTIFACT_DIR = "Artifacts"
FILE_NAME = "NSE-Tata_Global_Beverages_Limited.csv"

TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR = os.path.join("saved_model")
MODEL_FILE_NAME  :str = "model.pkl"

"""
DataIngestion related constant start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME:str =  "StockPredictor"
DATA_INGESTION_DATABASE_NAME:str =  "DINESH"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT: float = 0.2  

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME

"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME:str = "preprocessing.pkl"

