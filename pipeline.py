from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def build_ml_pipeline() -> Pipeline:
    """Builds a complete, leak-free Spark ML feature and estimator pipeline."""
    categorical_cols = ["device_type", "location_country", "merchant_category"]
    numeric_cols = [
        "amount", 
        "velocity_last_24h", 
        "amount_to_avg_ratio", 
        "hour_of_day", 
        "is_night_txn", 
        "is_high_risk_amount"
    ]
    
    # 1. String Indexing for Categorical columns
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in categorical_cols
    ]
    
    # 2. One-Hot Encoding
    encoders = [
        OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_vec")
        for col in categorical_cols
    ]

    # 3. Assemble features
    assembler_inputs = [f"{col}_vec" for col in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features")
    
    # 4. Standard Scaler
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
    
    # 5. Gradient-Boosted Trees Classifier
    gbt = GBTClassifier(
        labelCol="is_fraud", 
        featuresCol="features", 
        maxIter=15, 
        seed=42
    )
    
    stages = indexers + encoders + [assembler, scaler, gbt]
    return Pipeline(stages=stages)

def train_and_evaluate(df: DataFrame, model_save_path: str = None) -> PipelineModel:
    """Splits data, trains the GBT pipeline, and outputs evaluation metrics."""
    train_df, test_df = df.randomSplit([0.75, 0.25], seed=123)
    
    pipeline = build_ml_pipeline()
    evaluator_roc = BinaryClassificationEvaluator(
        labelCol="is_fraud", 
        rawPredictionCol="rawPrediction", 
        metricName="areaUnderROC"
    )
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="is_fraud", 
        rawPredictionCol="rawPrediction", 
        metricName="areaUnderPR"
    )

    print("[*] Training GBT Model on Train Dataset...")
    model = pipeline.fit(train_df)
    
    predictions = model.transform(test_df)
    
    roc_score = evaluator_roc.evaluate(predictions)
    pr_score = evaluator_pr.evaluate(predictions)
    
    print(f"\n================ MODEL EVALUATION ================")
    print(f"Area Under ROC (ROC-AUC) : {roc_score:.4f}")
    print(f"Area Under PR (PR-AUC)   : {pr_score:.4f}")
    print(f"==================================================\n")
    
    if model_save_path:
        model.write().overwrite().save(model_save_path)
        print(f"[+] Model artifact successfully exported to: {model_save_path}")
        
    return model