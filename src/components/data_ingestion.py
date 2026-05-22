'''import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from zipfile import Path
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config= DataIngestionConfig()
        self.utils = MainUtils()

    def export_collection_as_dataframe(self,collection_name,db_name):

        try:
            mongo_client = MongoClient(MONGO_DB_URL)

            collection = mongo_client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=['_id'],axis=1)
            
            df.replace({"na":np.nan},inplace=True)

            return df
        except Exception as e:
            raise CustomException(e,sys)

    def export_data_into_feature_store_file_path(self)-> pd.DataFrame:

        try:

            logging.info(f"Exporting data from mongodb")
            raw_file_path = self.data_ingestion_config.artifact_folder

            os.makedirs(raw_file_path,exist_ok=True)

            sensor_data = self.export_collection_as_dataframe(
                collection_name= MONGO_COLLECTION_NAME,
                db_name = MONGO_DATABASE_NAME
            )

            logging.info(f"saving exported data into feature store file path :{raw_file_path}")

            feature_store_file_path = os.path.join(raw_file_path,'wafer_fault.csv')

            sensor_data.to_csv(feature_store_file_path,index=False)

            return feature_store_file_path
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_ingestion(self) -> Path:

        logging.info("Entered initiated_data_ingestion method of data_integration class")

        try:
            feature_store_file_path = self.export_data_into_feature_store_file_path()

            logging.info("got the data from mongodb")

            logging.info("exited initiate_data_ingestion methos of data ingestion class")

            return feature_store_file_path
        except Exception as e:
            raise CustomException(e,sys) from e'''

#new code i have top put in given by chat gpt
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.constant import artifact_folder, DATA_FILE_PATH, TARGET_COLUMN
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    feature_store_file_path: str = os.path.join(
        artifact_folder, "feature_store", "wafer.csv"
    )
    train_file_path: str = os.path.join(
        artifact_folder, "dataset", "train.csv"
    )
    test_file_path: str = os.path.join(
        artifact_folder, "dataset", "test.csv"
    )


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion using local CSV")

            df = pd.read_csv(DATA_FILE_PATH)
            logging.info(f"Dataset loaded successfully with shape {df.shape}")

            if "Good/Bad" in df.columns and TARGET_COLUMN not in df.columns:
                df.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)

            if TARGET_COLUMN not in df.columns:
                raise Exception(f"Target column '{TARGET_COLUMN}' not found in dataset")

            print("Full dataset target unique values:", df[TARGET_COLUMN].unique())

            os.makedirs(
                os.path.dirname(self.data_ingestion_config.feature_store_file_path),
                exist_ok=True
            )

            df.to_csv(
                self.data_ingestion_config.feature_store_file_path,
                index=False
            )
            logging.info("Feature store file saved")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df[TARGET_COLUMN]
            )

            os.makedirs(
                os.path.dirname(self.data_ingestion_config.train_file_path),
                exist_ok=True
            )

            train_set.to_csv(
                self.data_ingestion_config.train_file_path,
                index=False
            )
            test_set.to_csv(
                self.data_ingestion_config.test_file_path,
                index=False
            )

            logging.info("Train and test files saved successfully")

            return (
                self.data_ingestion_config.train_file_path,
                self.data_ingestion_config.test_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)