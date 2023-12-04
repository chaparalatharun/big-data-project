from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col


def evaluate_linear_regression(df, target_col, feature_cols):
    spark = SparkSession.builder.appName("LinearRegression").getOrCreate()
    sdf = spark.createDataFrame(df)
    trainingData, testData = sdf.randomSplit([0.7, 0.3])

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")


    trainingData = assembler.transform(trainingData)
    testData = assembler.transform(testData)

    lr = LinearRegression(featuresCol="features", labelCol=target_col)


    model = lr.fit(trainingData)


    predictions = model.transform(testData)


    TSS = sdf.select(col(target_col)).rdd.map(lambda x: x[0]).variance() * (sdf.count() - 1)
    RSS = predictions.select(col(target_col), col("prediction")).rdd.map(lambda x: (x[0] - x[1])**2).sum()


    
    R_squared =1 - (RSS / TSS)


    coefficients = model.coefficients
    intercept = model.intercept
    print("Coefficients: ")
    feature_names = assembler.getInputCols()
    for i in range(len(coefficients)):
        print(feature_names[i], ": ", coefficients[i], " (p-value: ", model.summary.pValues[i], ")")
    print("Intercept: ", intercept)


    evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction")
    mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("TSS: ", TSS)
    print("RSS: ", RSS)
    print("R-squared: ", R_squared)
