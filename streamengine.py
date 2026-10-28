from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, 
    DoubleType, IntegerType
)
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel

# Schema for incoming JSON payload
STREAM_SCHEMA = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("timestamp", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("device_type", StringType(), True),
    StructField("location_country", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("merchant_category", StringType(), True),
    StructField("velocity_last_24h", IntegerType(), True)
])

def run_in_memory_streaming_inference(
    spark: SparkSession, 
    model_path: str,
    source_dir: str
):
    """
    Demonstrates Spark Structured Streaming reading JSON events from a folder,
    transforming features, and scoring records in real-time.
    """
    model = PipelineModel.load(model_path)
    
    streaming_df = (
        spark.readStream
        .schema(STREAM_SCHEMA)
        .json(source_dir)
    )
    
    # Feature engineering for stream
    processed_stream = (
        streaming_df
        .withColumn("timestamp", F.to_timestamp(F.col("timestamp")))
        .withColumn("hour_of_day", F.hour(F.col("timestamp")))
        .withColumn("is_night_txn", 
                    F.when((F.col("hour_of_day") >= 23) | (F.col("hour_of_day") <= 5), 1).otherwise(0))
        .withColumn("is_high_risk_amount", 
                    F.when(F.col("amount") >= 3000.0, 1).otherwise(0))
        .withColumn("amount_to_avg_ratio", F.lit(1.2)) # Real-time default fallback
        .withColumn("is_fraud", F.lit(0)) # Placeholder label for pipeline execution
    )
    
    scored_stream = model.transform(processed_stream)
    
    # Select critical alerts
    fraud_alerts = (
        scored_stream
        .select("transaction_id", "user_id", "amount", "prediction", "probability")
        .filter(F.col("prediction") == 1.0)
    )

    # Select critical alerts
    fraud_alerts = (
        scored_stream
        .select("transaction_id", "user_id", "amount", "prediction", "probability")
        .filter(F.col("prediction") == 1.0)
    )
    
    # In-memory query sink for demonstration
    query = (
        fraud_alerts.writeStream
        .format("memory")
        .queryName("fraud_alerts_sink")
        .outputMode("append")
        .start()
    )
    
    return query