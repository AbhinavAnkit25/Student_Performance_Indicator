import os
import sys

from dataclasses import dataclass

from src.execption import CustomException
from src.logger import logging

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
     mean_squared_error
     )
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoost

from src.utlis import (
    save_object,
    evaluate_model
)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array, test_array,):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "CatBoost Regressor": CatBoost(),
                "K-Neighbour Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor()
            }
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    ##'splitter': ['best' , 'random'],
                    ##'max_features' : ['sqrt', 'log2'],
                },
                "Random Forest": {
                    ##'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    ##'max_features' : ['sqrt', 'log2', None],
                    'n_estimators': [8,16,32,64,128,256],
                },
                "Gradient Boosting": {
                    ##'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    ##'criterion': ['squared_error', 'friedman_mse'],
                    'n_estimators': [8,16,32,64,128,256],
                },
                "Linear Regression": {
                    'fit_intercept' : [True , False],
                },
                "AdaBoost Regressor":{
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate':[0.1,0.01,0.05,0.001,1.0],
                    ##'loss':['linear', 'square', 'exponential'],
                },
                "CatBoost Regressor":{
                    'iterations': [100,200,300,500],
                    'learning_rate':[0.01,0.05,0.1,0.2],
                    'depth':[4,6,8,10],
                    ##'l2_leaf_reg':[1,3,5,7,9],
                    ##'border_count':[32,64,128],
                },
                "K-Neighbour Regressor":{
                    'n_neighbors': [3,5,7,9],
                    ##'weights':['uniform', 'distance'],
                    ##'algorithm':['auto','ball_tree', 'kd_tree', 'brute'],
                    ##'leaf_size':[20,30,40],
                },
                "XGB Regressor":{
                    'n_estimators':[8,16,32,64,128,256],
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    ##'max_depth':[3,5,10,None],
                    ##'subsample':[0.6,0.7,0.75,0.8,0.85,0.9,1.0],
                    ##'colsample_bytree':[0.6,0.7,0.8,0.9,1.0],
                    ##'gamma':[0,0.1,0.2,0.3],
                    ##'reg_alpha':[0,0.1,0.5,1.0],
                    ##'reg_lambda':[1.0,1.5,2.0],
                },
            }

            model_report:dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test = y_test,
                models=models,
                param = params,
            )

            ## To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## TO get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both traning and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return (f"The best model is : {best_model_name} \nScore is : {best_model_score}")

        except Exception as e:
         raise CustomException(e,sys)