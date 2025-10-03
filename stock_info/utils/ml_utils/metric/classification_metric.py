from stock_info.entity.artifact_entity import RegressionMetricArtifact
from stock_info.Exception.exception import StockPredictionException
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import sys
def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    try:
        
        model_mae  = mean_absolute_error(y_true, y_pred)
        model_mse = mean_squared_error(y_true, y_pred)
        model_rmse = root_mean_squared_error(y_true, y_pred)
        model_r2 = r2_score(y_true, y_pred)
        
        regression_metric = RegressionMetricArtifact(model_mae=model_mae, model_mse=model_mse, model_rmse=model_rmse, model_r2=model_r2)
        return regression_metric
    except Exception as e:
        raise StockPredictionException(e, sys)