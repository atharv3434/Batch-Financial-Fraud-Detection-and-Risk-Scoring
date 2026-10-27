from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from src.config import TRANSACTION_SCHEMA

def ingest_raw_transactions(spark, source_path: str) -> DataFrame:
    """Reads raw CSV logs applying explicit schema validation."""
    return (
        spark.read
        .option("header", "true")
        .schema(TRANSACTION_SCHEMA)
        .csv(source_path)
    )

def enrich_and_clean_transactions(df: DataFrame) -> DataFrame:
    """
    Cleans data and derives windowed behavioral features:
    1. Imputes null values.
    2. Calculates user historical spending mean and deviation via window partitioning.
    3. Derives relative transaction deviation score: amount / (user_avg + 1).
    4. Extracts temporal features (hour_of_day, is_weekend).
    """
    # Deduplicate on transaction_id
    cleaned_df = df.dropDuplicates(["transaction_id"])
    
    # Impute missing velocity or amounts
    cleaned_df = cleaned_df.na.fill({
        "velocity_last_24h": 0,
        "amount": 0.0,
        "device_type": "Unknown",
        "location_country": "XX"
    })

        .partitionBy("location_country")
        .parquet(target_path)
    ) # Window partitioned by user to compute user baseline profile
    user_window = Window.partitionBy("user_id")
    
    transformed_df = (
        cleaned_df
        .withColumn("user_avg_amount", F.avg("amount").over(user_window))
        .withColumn("amount_to_avg_ratio", 
                    F.round(F.col("amount") / (F.col("user_avg_amount") + 1.0), 3))
        .withColumn("hour_of_day", F.hour(F.col("timestamp")))
        .withColumn("is_night_txn", 
                    F.when((F.col("hour_of_day") >= 23) | (F.col("hour_of_day") <= 5), 1).otherwise(0))
        .withColumn("is_high_risk_amount", 
                    F.when(F.col("amount") >= 3000.0, 1).otherwise(0))
    )
    
    return transformed_df

def write_silver_layer(df: DataFrame, target_path: str) -> None:
    """Persists enriched dataset to Parquet format partitioned by country."""
    (
        df.write
        .mode("overwrite")