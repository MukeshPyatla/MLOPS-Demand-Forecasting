import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, dayofweek, dayofmonth

def main(raw_data_path, processed_output_path):
    """
    Main ETL logic for feature engineering on sales data.
    """
    spark = SparkSession.builder.appName("DemandForecastingETL").getOrCreate()

    print(f"Reading raw data from {raw_data_path}")
    df = spark.read.csv(raw_data_path, header=True, inferSchema=True)

    # Convert Date string to Date type
    df_transformed = df.withColumn("Date", to_date(col("Date")))

    # Feature Engineering
    print("Performing feature engineering...")
    df_features = df_transformed.withColumn("Year", year(col("Date"))) \
                                .withColumn("Month", month(col("Date"))) \
                                .withColumn("DayOfWeek", dayofweek(col("Date"))) \
                                .withColumn("DayOfMonth", dayofmonth(col("Date")))

    # For forecasting, Prophet expects columns 'ds' (datestamp) and 'y' (value)
    # We will rename them here for convention. We also keep identifiers.
    df_final = df_features.select(
        col("Date").alias("ds"),
        col("StoreID"),
        col("ProductID"),
        col("Sales").alias("y")
    ).orderBy("StoreID", "ProductID", "ds")

    print(f"Writing processed data to {processed_output_path}")
    df_final.write.mode("overwrite").parquet(processed_output_path)
    
    print("ETL job completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, required=True, help="Path to raw input data")
    parser.add_argument("--processed_output_path", type=str, required=True, help="Path to store processed output data")
    args = parser.parse_args()
    
    main(args.raw_data_path, args.processed_output_path)