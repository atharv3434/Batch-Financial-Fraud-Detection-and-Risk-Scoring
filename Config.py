import os
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, 
    DoubleType, IntegerType, TimestampType
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "transactions.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "silver_transactions.parquet")
MODEL_ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts", "fraud_gbt_model")
STREAMING_CHECKPOINT_PATH = os.path.join(BASE_DIR, "artifacts", "checkpoints")

TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("timestamp", TimestampType(), True),
    StructField("user_id", StringType(), True),
    StructField("device_type", StringType(), True),
    StructField("location_country", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("merchant_category", StringType(), True),
    StructField("velocity_last_24h", IntegerType(), True),
    StructField("is_fraud", IntegerType(), True)
])

def get_spark_session(app_name: str = "FraudAnalyticsPlatform") -> SparkSession:
    """Creates a local, optimized SparkSession for analytics and ML."""
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )