import os
import sys
from src.config import (
    get_spark_session, 
    RAW_DATA_PATH, 
    PROCESSED_DATA_PATH, 
    MODEL_ARTIFACT_PATH
)
from src.batch_etl import ingest_raw_transactions, enrich_and_clean_transactions, write_silver_layer
from src.ml_pipeline import train_and_evaluate
from pyspark.sql import functions as F

def main():
    print("==================================================================")
    print("        PYSPARK DATA SCIENCE: REAL-TIME FRAUD DETECTION           ")
    print("==================================================================")
    
    spark = get_spark_session("FraudDetectionPlatformDriver")
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        # Phase 1: Ingestion & Feature Engineering (ETL)
        print("\n--- PHASE 1: BATCH ETL & WINDOWED FEATURE STORE ---")
        raw_df = ingest_raw_transactions(spark, RAW_DATA_PATH)
        print(f"[*] Ingested {raw_df.count()} records from Bronze tier.")
        
        silver_df = enrich_and_clean_transactions(raw_df)
        print("[*] Windowed features derived (amount_to_avg_ratio, is_night_txn, etc.).")
        silver_df.select(
            "transaction_id", "user_id", "amount", "user_avg_amount", "amount_to_avg_ratio", "is_fraud"
        ).show(5, truncate=False)

        write_silver_layer(silver_df, PROCESSED_DATA_PATH)
        print(f"[+] Persisted Silver Tier to {PROCESSED_DATA_PATH}")
        
        # Phase 2: ML Model Training & Validation
        print("\n--- PHASE 2: SPARK MLLIB MODEL TRAINING (GBT) ---")
        train_and_evaluate(silver_df, MODEL_ARTIFACT_PATH)
        
        # Phase 3: Model Inference Verification
        print("\n--- PHASE 3: BATCH INFERENCE & SCORING VERIFICATION ---")
        from pyspark.ml import PipelineModel
        loaded_model = PipelineModel.load(MODEL_ARTIFACT_PATH)
        sample_predictions = loaded_model.transform(silver_df)
        
        print("[*] Top High-Risk Fraud Alerts Identified:")
        sample_predictions.filter(F.col("prediction") == 1.0).select(
            "transaction_id", "user_id", "amount", "merchant_category", "prediction"
        ).show(5, truncate=False)

        print("\n==================================================================")
        print("                 PIPELINE RUN SUCCESSFULLY COMPLETED              ")
        print("==================================================================")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()