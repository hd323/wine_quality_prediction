import sys

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def clean_data(df):
    # Clean header and cast columns to double
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

"""Main function for the application"""
if __name__ == "__main__":
    
    # Create a Spark application
    spark = SparkSession.builder \
        .appName('wine_app') \
        .getOrCreate()

    # Create a Spark context to report logging information related to Spark
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    # Set default paths
    default_input_path = "s3://hd323-2/TrainingDataset.csv"
    default_valid_path = "s3://hd323-2/ValidationDataset.csv"
    default_output_path = "s3://hd323-2/testmodel.model"

    # Override paths if command line arguments are provided
    if len(sys.argv) == 4:
        input_path, valid_path, output_path = sys.argv[1:4]
        output_path += "testmodel.model"
    else:
        input_path, valid_path, output_path = default_input_path, default_valid_path, default_output_path

    # Read CSV files into DataFrames
    training_df = (spark.read
                   .format("csv")
                   .option('header', 'true')
                   .option("sep", ";")
                   .option("inferschema", 'true')
                   .load(input_path))

    training_data_set = clean_data(training_df)

    validation_df = (spark.read
                     .format("csv")
                     .option('header', 'true')
                     .option("sep", ";")
                     .option("inferschema", 'true')
                     .load(valid_path))

    validation_data_set = clean_data(validation_df)

    # Define required features for training
    selected_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'chlorides', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol', 'quality']

    # Create a vector column named 'features' using selected features
    feature_assembler = VectorAssembler(inputCols=selected_features, outputCol='features')

    # Create a StringIndexer for the 'quality' column
    label_indexer = StringIndexer(inputCol="quality", outputCol="label")

    # Cache data for faster access
    training_data_set.cache()
    validation_data_set.cache()

    # Set up RandomForestClassifier with specified parameters
    random_forest_classifier = RandomForestClassifier(labelCol='label', 
                                                      featuresCol='features',
                                                      numTrees=150,
                                                      maxBins=8, 
                                                      maxDepth=15,
                                                      seed=150,
                                                      impurity='gini')

    # Create a pipeline for the classification
    classification_pipeline = Pipeline(stages=[feature_assembler, label_indexer, random_forest_classifier])
    trained_model = classification_pipeline.fit(training_data_set)

    # Validate the trained model on the test data
    predictions = trained_model.transform(validation_data_set)

    # Evaluate the model's accuracy
    results = predictions.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(labelCol='label', 
                                                  predictionCol='prediction', 
                                                  metricName='accuracy')

    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy = ', accuracy)

    # Calculate and print the weighted f1 score
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted f1 score = ', metrics.weightedFMeasure())

    # Retrain the model on multiple parameters using CrossValidator
    cv_model = None
    param_grid = ParamGridBuilder() \
        .addGrid(random_forest_classifier.maxBins, [9, 8, 4]) \
        .addGrid(random_forest_classifier.maxDepth, [25, 6, 9]) \
        .addGrid(random_forest_classifier.numTrees, [500, 50, 150]) \
        .addGrid(random_forest_classifier.minInstancesPerNode, [6]) \
        .addGrid(random_forest_classifier.seed, [100, 200, 5043, 1000]) \
        .addGrid(random_forest_classifier.impurity, ["entropy", "gini"]) \
        .build()

    # Create a new pipeline for cross-validation
    cross_validator = CrossValidator(estimator=classification_pipeline,
                                     estimatorParamMaps=param_grid,
                                     evaluator=evaluator,
                                     numFolds=2)

    # Fit the CrossValidator to the training data
    cv_model = cross_validator.fit(training_data_set)

    # Save the best model to a new variable `final_model`
    final_model = cv_model.bestModel
    print(final_model)

    # Print accuracy of the best model on the validation data
    predictions = final_model.transform(validation_data_set)
    results = predictions.select(['prediction', 'label'])
    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy1 = ', accuracy)

    # Calculate and print the weighted f1 score for the best model
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted f1 score = ', metrics.weightedFMeasure())

    # Save the best model to the specified output path
    best_model_path = output_path
    final_model.write().overwrite().save(best_model_path)
    sys.exit(0)
