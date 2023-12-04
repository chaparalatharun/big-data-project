from pyspark_linear_regression_evaluator import evaluate_linear_regression

from sklearn.datasets import fetch_california_housing
import pandas as pd
california_housing = fetch_california_housing()
california_df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
california_df['Target'] = california_housing.target
evaluate_linear_regression(california_df,"Target",["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup"])