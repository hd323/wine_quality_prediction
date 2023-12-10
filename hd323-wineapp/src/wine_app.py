import os
import sys

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

def clean_data(df):
    # Clean header
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

"""Main function for application"""
if __name__ == "__main__":
    
    # Create a Spark application
    spark = SparkSession.builder \
        .appName('wine_app') \
        .getOrCreate()

    # Create a Spark context to report logging information related to Spark
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    current_dir = os.getcwd()
    # Load and parse the data file into an RDD of LabeledPoint.
    if len(sys.argv) > 3:
        sys.exit(-1)
    elif len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        if not ("/" in input_path):
            input_path = os.path.join(current_dir, input_path)
        model_path = os.path.join(current_dir, "testdata.model")
        print("----Input file for test data is---")
        print(input_path)
    else:
         
        print("-----------------------")
        print(current_dir)
        input_path = os.path.join(current_dir, "testdata.csv")
        model_path = os.path.join(current_dir, "testdata.model")

    # Read CSV file into a DataFrame
    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load(input_path))
    
    df_cleaned = clean_data(df)

    # Define required features
    required_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    
    # Load the trained model
    model = PipelineModel.load(model_path)
    
    # Make predictions on the cleaned data
    predictions = model.transform(df_cleaned)
    print(predictions.show(5))
    
    # Evaluate the model accuracy
    results = predictions.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(
                                            labelCol='label', 
                                            predictionCol='prediction', 
                                            metricName='accuracy')

    # Print the accuracy of the model
    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy = ', accuracy)
    
    # Compute and print the weighted F1 score
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted F1 Score = ', metrics.weightedFMeasure())
    
    sys.exit(0)
