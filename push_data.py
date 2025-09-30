# from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi
# uri = "mongodb+srv://dinu08642_db_user:Dinesh4953@cluster0.vcjat5x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# # Create a new client and connect to the server
# client = MongoClient(uri, server_api=ServerApi('1'))

# # Send a ping to confirm a successful connection
# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)

import os
import sys
import json

from dotenv  import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from stock_info.Exception.exception import StockPredictionException
from stock_info.logging.logger import logging

class StockDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise StockPredictionException(e, sys)

    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise StockPredictionException(e, sys)
        
    def insert_data_to_mongodb(self, records, database, collection):
        try:
            self.database = database
            self.records = records
            self.collection = collection
            
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return (len(self.records))
        
        except Exception as e:
            return StockPredictionException(e, sys)

if __name__ == "__main__":
    FILE_PATH = r"stock_data\NSE-Tata_Global_Beverages_Limited.csv"
    DATABASE = "DINESH"
    Collection = "StockPredictor"
    stockkobj  = StockDataExtract()
    records = stockkobj.csv_to_json_converter(file_path=FILE_PATH)
    # print(records)
    no_of_records = stockkobj.insert_data_to_mongodb(records, DATABASE, Collection)
    print(no_of_records)
