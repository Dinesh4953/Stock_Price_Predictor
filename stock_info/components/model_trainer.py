import os
import sys

from stock_info.Exception.exception import StockPredictionException
from stock_info.logging.logger import logging

from stock_info.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, RegressionMetricArtifact
from stock_info.entity.config_entity import ModelTrainerConfig

from stock_info.utils.mainutils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from stock_info.utils.ml_utils.model.estimator import StockModel
from stock_info.utils.ml_utils.metric.classification_metric import get_regression_score


from sklearn.linear_model import (LinearRegression,
                                  SGDRegressor, 
                                  Ridge, 
                                  Lasso,
                                  ElasticNet,
                                  BayesianRidge,
                                  ARDRegression,
                                  PassiveAggressiveRegressor,
                                  HuberRegressor,
                                  RANSACRegressor,
                                  TheilSenRegressor,
                                  QuantileRegressor)
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import(
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
    VotingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
 
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise StockPredictionException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Linear Regression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_jobs=-1),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(),
            "BayesianRidge": BayesianRidge(),
            "ARDRegression":ARDRegression(),
            "SGDRegressor":SGDRegressor(),
            "PassiveAggressiveRegressor":PassiveAggressiveRegressor(),
            "HuberRegressor":HuberRegressor(),
            "RANSACRegressor":RANSACRegressor(),
            "TheilSenRegressor":TheilSenRegressor(n_jobs=-1),
            "QuantileRegressor":QuantileRegressor(),
            "DecisionTreeRegressor":DecisionTreeRegressor(),
            "GradientBoostingRegressor":GradientBoostingRegressor(),
            "AdaBoostRegressor":AdaBoostRegressor(),
            "ExtraTreesRegressor":ExtraTreesRegressor(n_jobs=-1),
            "BaggingRegressor":BaggingRegressor(n_jobs=-1),
            "HistGradientBoostingRegressor":HistGradientBoostingRegressor(),
            "KNeighborsRegressor":KNeighborsRegressor(n_jobs=-1),
            # "GaussianProcessRegressor":GaussianProcessRegressor(),
            "PLSRegression":PLSRegression()
        }
        params = {
            "SVR": {
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1, 0.2],
        "gamma": ['scale', 'auto']
    },
            "Linear Regression":{
                "fit_intercept": [True, False],
                "positive": [True, False]
            },
            "Ridge":{
                'alpha': [0.01, 0.1, 1, 10], 'max_iter': [1000, 5000, 10000],
                "solver":["auto", "svd", "lsqr", "saga"],
                "fit_intercept":[True, False]
            },
            "RandomForestRegressor":{
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            },
            "Lasso":{
                'alpha': [0.01, 0.1, 1, 10], 'max_iter': [1000, 5000, 10000],
                "fit_intercept":[True, False],
                "selection":['cyclic', 'random']
            },
            "ElasticNet":{
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "l1_ratio":[0.1, 0.5, 0.9],
                "fit_intercept":[True, False],
                "max_iter":[1000, 5000],
                "selection":['cyclic', 'random']
            },
            "DecisionTreeRegressor":{
                "criterion": ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
            },
            "SGDRegressor":{
                "loss": ['squared_error','huber','epsilon_insensitive'],
                "penalty": ['l2', 'l1','elasticnet'],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
                "learning_rate": ['constant', 'optimal', 'invscaling', 'adaptive'],
                "eta0":[0.001, 0.01, 0.1],
                "max_iter":[1000, 5000]
            },
            "BayesianRidge":{
                # "n_iter":[100, 300, 500],
                "tol":[1e-4, 1e-3],
                "alpha_1":[1e-6, 1e-5],
                "alpha_2":[1e-6, 1e-5],
                "lambda_1": [1e-6, 1e-5],
                "lambda_2": [1e-6, 1e-5]
            },
            "ARDRegression": {
        "tol": [1e-4, 1e-3],
        "alpha_1": [1e-6, 1e-5],
        "alpha_2": [1e-6, 1e-5],
        "lambda_1": [1e-6, 1e-5],
        "lambda_2": [1e-6, 1e-5]
    },
            "PassiveAggressiveRegressor": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "fit_intercept": [True, False],
        "max_iter": [1000, 5000],
        "tol": [1e-4, 1e-3]
    },
            "HuberRegressor": {
        "epsilon": [1.1, 1.35, 1.5],
        "alpha": [0.0001, 0.001, 0.01],
        "max_iter": [100, 500],
        "fit_intercept": [True, False]
    },
    "RANSACRegressor": {
        "min_samples": [0.5, 0.75, 1.0],
        "residual_threshold": [1, 5, 10],
        "max_trials": [50, 100, 200]
    },
    "TheilSenRegressor": {
        "max_subpopulation": [1000],
        
    },
    "QuantileRegressor": {
        "quantile": [0.25, 0.5, 0.75],
        "alpha": [0.0, 0.01, 0.1],
        # "max_iter": [500, 1000]
    },
    "GradientBoostingRegressor": {
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "n_estimators": [50, 100, 200],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "AdaBoostRegressor": {
        "learning_rate": [0.001, 0.01, 0.1, 0.5, 1.0],
        "n_estimators": [50, 100, 200, 500]
    },
    "ExtraTreesRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ['auto', 'sqrt', 'log2']
    },
    "BaggingRegressor": {
        "n_estimators": [10, 50, 100],
        "max_samples": [0.5, 0.75, 1.0],
        "max_features": [0.5, 0.75, 1.0]
    },
    "HistGradientBoostingRegressor": {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_iter": [100, 200, 500],
        "max_depth": [None, 3, 5, 10],
        "min_samples_leaf": [20, 30, 50]
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ['uniform', 'distance'],
        "p": [1, 2]
    },
    "PLSRegression": {
        "n_components": [1, 2, 5, 10],
        "scale": [True, False]
    }
    
        }
        model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test = X_test, y_test=y_test,
                                            models=models, params=params)
        best_model_score = max(sorted(model_report.values()))
        
        
        ## To get best model name from dict

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        print("-------------------------------------------------")
        print(best_model)
        print("-------------------------------------------------")
        
        
        y_train_pred = best_model.predict(X_train)
        regression_train_metric = get_regression_score(y_true=y_train, y_pred=y_train_pred)
        
        y_test_pred = best_model.predict(X_test)
        regression_test_metric = get_regression_score(y_true=y_test, y_pred=y_test_pred)
        
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)
        
        stockModel = StockModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=stockModel)
        
        save_object("final_model/model.pkl", best_model)
        
        ## Model Trainer Artifact
        
        model_trainer_artifact = ModelTrainerArtifact(   
                             trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=regression_train_metric,
                             test_metric_artifact=regression_test_metric
                             ) 
        logging.info(f"Model trainder artifact: {model_trainer_artifact}")
        return model_trainer_artifact
    
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            ## loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            x_train, y_train, x_test, y_test = [
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            ]
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact
        except Exception as e:
            raise StockPredictionException(e, sys) 

        
        