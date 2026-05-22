'''import sys
from typing import Generator, List, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder= os.path.join(artifact_folder)
    trained_model_path= os.path.join(artifact_folder,"model.pkl" )
    expected_accuracy=0.45
    model_config_file_path= os.path.join('config','model.yaml')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.models = {
                        'XGBClassifier': XGBClassifier(),
                        'GradientBoostingClassifier' : GradientBoostingClassifier(),
                        'SVC' : SVC(),
                        'RandomForestClassifier': RandomForestClassifier()
                        }
    def evaluate_models(self, X, y, models):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            report = {}
            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train, y_train)  # Train model
                y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)
                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)
                report[list(models.keys())[i]] = test_model_score
            return report
        except Exception as e:
            raise CustomException(e, sys)
    def get_best_model(self,
                    x_train:np.array,
                    y_train: np.array,
                    x_test:np.array,
                    y_test: np.array):
        try:
            model_report: dict = self.evaluate_models(
                 x_train =  x_train,
                 y_train = y_train,
                 x_test =  x_test,
                 y_test = y_test,
                 models = self.models
            )
            print(model_report)
            best_model_score = max(sorted(model_report.values()))
            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model_object = self.models[best_model_name]
            return best_model_name, best_model_object, best_model_score
        except Exception as e:
            raise CustomException(e,sys)
       
    def finetune_best_model(self,
                            best_model_object:object,
                            best_model_name,
                            X_train,
                            y_train,
                            ) -> object: 
        try:
            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]
            grid_search = GridSearchCV(
                best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1 )
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            print("best params are:", best_params)
            finetuned_model = best_model_object.set_params(**best_params)
            return finetuned_model
       
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"Splitting training and testing input and target feature")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            logging.info(f"Extracting model config file path")
            logging.info(f"Extracting model config file path")
            model_report: dict = self.evaluate_models(X=x_train, y=y_train, models=self.models)
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = self.models[best_model_name]
            best_model = self.finetune_best_model(
                best_model_name= best_model_name,
                best_model_object= best_model,
                X_train= x_train,
                y_train= y_train
            )
            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)
            print(f"best model name {best_model_name} and score: {best_model_score}")
            if best_model_score < 0.5:
                raise Exception("No best model found with an accuracy greater than the threshold 0.6")
            logging.info(f"Best found model on both training and testing dataset")
            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_path}"
            )
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            return self.model_trainer_config.trained_model_path
        except Exception as e:
            raise CustomException(e, sys)'''



#new code i have top put in given by chat gpt
'''import sys
import os
import numpy as np

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join(artifact_folder)
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join("config", "model.yaml")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

        self.models = {
            "XGBClassifier": XGBClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC(),
            "RandomForestClassifier": RandomForestClassifier()
        }

    def evaluate_models(self, x_train, y_train, x_test, y_test, models):
        try:
            report = {}

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                model.fit(x_train, y_train)

                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)

                logging.info(f"{model_name} train accuracy: {train_model_score}")
                logging.info(f"{model_name} test accuracy: {test_model_score}")

                report[model_name] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model(
        self,
        x_train: np.array,
        y_train: np.array,
        x_test: np.array,
        y_test: np.array
    ):
        try:
            model_report: dict = self.evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=self.models
            )

            print(model_report)

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model_object = self.models[best_model_name]

            return best_model_name, best_model_object, best_model_score

        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(
        self,
        best_model_object: object,
        best_model_name: str,
        X_train,
        y_train,
    ) -> object:
        try:
            model_config = self.utils.read_yaml_file(
                self.model_trainer_config.model_config_file_path
            )

            model_param_grid = model_config["model_selection"]["model"][best_model_name]["search_param_grid"]

            grid_search = GridSearchCV(
                estimator=best_model_object,
                param_grid=model_param_grid,
                cv=5,
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            print("best params are:", best_params)

            finetuned_model = best_model_object.set_params(**best_params)

            return finetuned_model

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input and target feature")

            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info("Finding best base model")

            best_model_name, best_model, best_model_score = self.get_best_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test
            )

            logging.info(f"Best base model found: {best_model_name}")
            logging.info(f"Best base model score: {best_model_score}")

            logging.info("Fine-tuning best model")

            best_model = self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                X_train=x_train,
                y_train=y_train
            )

            best_model.fit(x_train, y_train)

            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)

            print(f"best model name {best_model_name} and score: {best_model_score}")

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"No best model found with an accuracy greater than the threshold {self.model_trainer_config.expected_accuracy}"
                )

            logging.info("Best found model on both training and testing dataset")
            logging.info(f"Saving model at path: {self.model_trainer_config.trained_model_path}")

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_path),
                exist_ok=True
            )

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)'''





#given by

import sys
import os
import numpy as np

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    artifact_folder = artifact_folder
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join("config", "model.yaml")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

        self.models = {
            "XGBClassifier": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                base_score=0.5,
                use_label_encoder=False
            ),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC(),
            "RandomForestClassifier": RandomForestClassifier()
        }

    def evaluate_models(self, x_train, y_train, x_test, y_test, models):
        try:
            report = {}

            y_train = np.array(y_train).astype(int)
            y_test = np.array(y_test).astype(int)

            logging.info(f"Unique classes in y_train: {np.unique(y_train)}")
            logging.info(f"Unique classes in y_test: {np.unique(y_test)}")

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                model.fit(x_train, y_train)

                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)

                logging.info(f"{model_name} train accuracy: {train_model_score}")
                logging.info(f"{model_name} test accuracy: {test_model_score}")

                report[model_name] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model(
        self,
        x_train: np.array,
        y_train: np.array,
        x_test: np.array,
        y_test: np.array
    ):
        try:
            model_report = self.evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=self.models
            )

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model_object = self.models[best_model_name]

            return best_model_name, best_model_object, best_model_score

        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(
        self,
        best_model_object: object,
        best_model_name: str,
        X_train,
        y_train,
    ) -> object:
        try:
            model_config = self.utils.read_yaml_file(
                self.model_trainer_config.model_config_file_path
            )

            model_param_grid = model_config["model_selection"]["model"][best_model_name]["search_param_grid"]

            grid_search = GridSearchCV(
                estimator=best_model_object,
                param_grid=model_param_grid,
                cv=5,
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            print("best params are:", best_params)

            finetuned_model = best_model_object.set_params(**best_params)

            return finetuned_model

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input and target feature")

            x_train = train_array[:, :-1]
            y_train = train_array[:, -1].astype(int)
            x_test = test_array[:, :-1]
            y_test = test_array[:, -1].astype(int)

            logging.info(f"y_train unique values: {np.unique(y_train)}")
            logging.info(f"y_test unique values: {np.unique(y_test)}")

            if len(np.unique(y_train)) < 2:
                raise Exception("Training labels contain only one class. Need both classes for classification.")

            best_model_name, best_model, best_model_score = self.get_best_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test
            )

            logging.info(f"Best base model found: {best_model_name}")
            logging.info(f"Best base model score: {best_model_score}")

            best_model = self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                X_train=x_train,
                y_train=y_train
            )

            best_model.fit(x_train, y_train)

            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)

            print(f"best model name {best_model_name} and score: {best_model_score}")

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"No best model found with an accuracy greater than the threshold {self.model_trainer_config.expected_accuracy}"
                )

            logging.info(f"Saving model at path: {self.model_trainer_config.trained_model_path}")

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_path),
                exist_ok=True
            )

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)