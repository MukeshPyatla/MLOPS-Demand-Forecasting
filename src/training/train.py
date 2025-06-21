import argparse
import os
import pandas as pd
from prophet import Prophet
import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def train_and_register_model(processed_data_path, model_name):
    """
    Trains a Prophet model for each store-product combination
    and registers the best one (for demonstration).
    In a real scenario, you'd register a model per combination.
    """
    mlflow.autolog()

    # --- MLFlow and Azure ML Setup ---
    # Using environment variables set by Azure ML job
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZUREML_WORKSPACE_NAME"],
    )

    print(f"Reading processed data from: {processed_data_path}")
    # Data is passed as a mounted input path
    all_files = [os.path.join(processed_data_path, f) for f in os.listdir(processed_data_path) if f.endswith('.parquet')]
    df = pd.concat((pd.read_parquet(f) for f in all_files), ignore_index=True)
    
    print("Data loaded successfully. Columns:", df.columns.tolist())
    print("Data types:", df.dtypes)

    # For this example, we'll train a model on one specific combination
    # A full-scale solution would loop through all combinations
    target_store = 'Store_1'
    target_product = 'Product_1'
    
    print(f"Filtering data for {target_store} and {target_product}")
    df_single_series = df[(df['StoreID'] == target_store) & (df['ProductID'] == target_product)].copy()
    
    if df_single_series.empty:
        raise ValueError("No data found for the selected store/product combination.")

    # Prophet requires 'ds' and 'y' columns. They should already be named correctly.
    df_train = df_single_series[['ds', 'y']]

    print("Training Prophet model...")
    # --- Model Training ---
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    model.fit(df_train)
    
    print("Model training complete.")

    # --- Model Evaluation (Example) ---
    future = model.make_future_dataframe(periods=90) # Forecast 90 days ahead
    forecast = model.predict(future)
    
    # Example metric: Log Mean Absolute Error of the forecast in the last 30 days of historical data
    df_eval = pd.merge(df_train, forecast[['ds', 'yhat']], on='ds')
    df_eval = df_eval.tail(30)
    mae = abs(df_eval['y'] - df_eval['yhat']).mean()
    print(f"Forecast MAE on last 30 days: {mae}")
    mlflow.log_metric("mae_forecast_last_30_days", mae)
    
    # --- Model Registration ---
    print(f"Registering model '{model_name}'")
    mlflow.prophet.log_model(
        pr_model=model,
        artifact_path="prophet-model",
        registered_model_name=model_name
    )
    
    print("Model registered successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data", type=str, required=True, help="Path to processed data folder.")
    parser.add_argument("--model_name", type=str, default="demand_forecasting_model", help="Name of the model to register.")
    args = parser.parse_args()
    
    train_and_register_model(args.processed_data, args.model_name)