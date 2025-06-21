import pandas as pd
import numpy as np
import os

def generate_synthetic_data(output_path='data/raw_sales_data.csv', years=3):
    """
    Generates synthetic daily sales data for multiple stores and products
    and saves it as a CSV file.
    """
    if os.path.exists(output_path):
        print(f"Data already exists at {output_path}. Skipping generation.")
        return

    print("Generating synthetic sales data...")
    num_days = 365 * years
    start_date = pd.to_datetime('2022-01-01')
    date_range = pd.date_range(start_date, periods=num_days, freq='D')

    stores = [f'Store_{i}' for i in range(1, 6)]      # 5 stores
    products = [f'Product_{j}' for j in range(1, 11)] # 10 products

    data = []
    for date in date_range:
        for store in stores:
            for product in products:
                # Base sales
                base_sales = 50 + (ord(store[-1]) % 5) * 10 + (ord(product[-1]) % 10) * 5

                # Weekly seasonality (higher sales on weekends)
                weekly_seasonality = 1.5 if date.dayofweek >= 5 else 1.0

                # Monthly seasonality (dip mid-month)
                monthly_seasonality = 1.0 - 0.2 * np.sin(2 * np.pi * date.day / 30.5)

                # Yearly seasonality (peak in December)
                yearly_seasonality = 1.0 + 0.5 * np.sin(2 * np.pi * (date.dayofyear - 80) / 365.25) + \
                                     (0.8 if date.month == 12 else 0)

                # Upward trend
                trend = 1.0 + (date - start_date).days / (num_days * 2)

                # Noise
                noise = np.random.normal(1.0, 0.1)

                sales = int(base_sales * weekly_seasonality * monthly_seasonality * yearly_seasonality * trend * noise)
                sales = max(0, sales) # Ensure sales are non-negative

                data.append([date, store, product, sales])

    df = pd.DataFrame(data, columns=['Date', 'StoreID', 'ProductID', 'Sales'])

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated and saved data to {output_path}")

if __name__ == '__main__':
    # This assumes you run the script from the root of the project directory
    generate_synthetic_data('data/raw_sales_data.csv')