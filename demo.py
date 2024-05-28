import json

# Create configurations for 10 products with sample values
configurations = {}

for i in range(10):
    product_id = f'product_{chr(65 + i)}'  # Generates product_A, product_B, ..., product_J
    configurations[product_id] = {
        'seasonality_mode': 'multiplicative' if i % 2 == 0 else 'additive',
        'yearly_seasonality': True if i % 3 == 0 else False,
        'weekly_seasonality': True,
        'daily_seasonality': False if i % 2 == 0 else True,
        'custom_seasonalities': [
            {'name': 'monthly', 'period': 30.5, 'fourier_order': 5 + i % 5},
            {'name': 'quarterly', 'period': 91.25, 'fourier_order': 10 + i % 10},
            {'name': 'bi-weekly', 'period': 14, 'fourier_order': 3 + i % 3}
        ]
    }

# Save configurations to a JSON file
with open('configurations.json', 'w') as config_file:
    json.dump(configurations, config_file, indent=4)

print("Configurations file created successfully.")

***********************************************************************************************

import json
import pandas as pd
from prophet import Prophet

# Load configurations from the JSON file
with open('configurations.json', 'r') as config_file:
    configurations = json.load(config_file)

# Example data setup: a dictionary of DataFrames, one for each product
data = {
    f'product_{chr(65 + i)}': pd.DataFrame({
        'ds': pd.date_range(start='2020-01-01', periods=100, freq='D'),
        'y': [i + (i**0.5) for i in range(100)]  # Example data
    }) for i in range(10)
}

# Dictionary to store trained models and forecasts
models = {}
forecasts = {}

# Training and forecasting for each product
for product, df in data.items():
    # Get the configuration for the current product
    config = configurations[product]
    
    # Initialize and configure the Prophet model
    model = Prophet(
        seasonality_mode=config['seasonality_mode'],
        yearly_seasonality=config['yearly_seasonality'],
        weekly_seasonality=config['weekly_seasonality'],
        daily_seasonality=config['daily_seasonality']
    )
    
    # Add custom seasonalities
    custom_seasonalities = config.get('custom_seasonalities', [])
    for seasonality in custom_seasonalities:
        model.add_seasonality(
            name=seasonality['name'],
            period=seasonality['period'],
            fourier_order=seasonality['fourier_order']
        )
    
    # Fit the model
    model.fit(df)
    
    # Create a DataFrame with future dates
    future = model.make_future_dataframe(periods=30)  # Forecasting 30 days into the future
    
    # Make the forecast
    forecast = model.predict(future)
    
    # Store the model and forecast
    models[product] = model
    forecasts[product] = forecast

# Example: Print the forecast for product_A
print(forecasts['product_A'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

***********************************************************************************************

import json

# Load configurations from the JSON file
with open('configurations.json', 'r') as config_file:
    configurations = json.load(config_file)

# Function to add a custom seasonality to a product's configuration
def add_custom_seasonality(product_id, name, period, fourier_order):
    if product_id in configurations:
        configurations[product_id]['custom_seasonalities'].append({
            'name': name,
            'period': period,
            'fourier_order': fourier_order
        })
        print(f"Added {name} seasonality to {product_id}.")
    else:
        print(f"Product {product_id} not found in configurations.")

# Function to remove a custom seasonality from a product's configuration
def remove_custom_seasonality(product_id, name):
    if product_id in configurations:
        custom_seasonalities = configurations[product_id]['custom_seasonalities']
        configurations[product_id]['custom_seasonalities'] = [
            s for s in custom_seasonalities if s['name'] != name
        ]
        print(f"Removed {name} seasonality from {product_id}.")
    else:
        print(f"Product {product_id} not found in configurations.")

# Add bi-monthly seasonality to product_A
add_custom_seasonality('product_A', 'bi-monthly', 60.5, 7)

# Remove bi-weekly seasonality from product_B
remove_custom_seasonality('product_B', 'bi-weekly')

# Save updated configurations back to the JSON file
with open('configurations.json', 'w') as config_file:
    json.dump(configurations, config_file, indent=4)

print("Updated configurations saved successfully.")

