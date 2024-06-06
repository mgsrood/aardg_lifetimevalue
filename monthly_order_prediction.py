import pandas_gbq
import json
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import datetime as dt
from dotenv import load_dotenv

# Load the variables from the .env file
load_dotenv()

# Get the GCP keys
gc_keys = os.getenv("AARDG_GOOGLE_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gc_keys

# Define full table id
features_project_id = os.getenv("UNNESTED_ORDER_DATA_PROJECT_ID")
features_dataset_id = os.getenv("UNNESTED_ORDER_DATA_DATASET_ID")
features_table_id = os.getenv("UNNESTED_ORDER_DATA_TABLE_ID")
features_full_table_id = f"{features_project_id}.{features_dataset_id}.{features_table_id}"

# SQL-query for features
unnested_query = f"SELECT * FROM `{features_full_table_id}`"

# Use GBQ to get data from BigQuery for features
unnested_df = pandas_gbq.read_gbq(unnested_query, project_id=features_project_id)
unnested_df['date_created'] = pd.to_datetime(unnested_df['date_created'])
unnested_df['year_month'] = unnested_df['date_created'].dt.strftime('%Y-%m')

# Select everything before february 2024
pre_february_df = unnested_df[unnested_df['date_created'] <= '2024-01-31']

# Only select pre march billing_emails for cust_month_tx
filtered_df = unnested_df[unnested_df['billing_email'].isin(pre_february_df['billing_email'])]

# Define the snapshot date
NOW = dt.datetime(2024, 2, 1).date()

# Calculate recency by subtracting current date from the latest InvoiceDate
features = pre_february_df.groupby('billing_email').agg({
  'date_created': lambda x: (NOW - x.max().date()).days,
  # Calculate frequency by counting unique number of invoices
  'order_id': pd.Series.nunique,
  # Calculate monetary value by summing all spend values
  'total': np.sum,
  # Calculate average and total quantity
  'quantity': ['mean', 'sum']}).reset_index()

# Rename the columns
features.columns = ['customer', 'recency', 'frequency', 'monetary', 'quantity_avg', 'quantity_total']

print(features)

# Build a pivot table counting invoices for each customer monthly
cust_month_tx = pd.pivot_table(data=filtered_df, values='order_id',
                               index=['billing_email'], columns=['year_month'],
                               aggfunc=pd.Series.nunique, fill_value=0)

# Store November 2011 data column name as a list
target = ['2024-02']

# Store target value as `Y`
Y = cust_month_tx[target]

# Store customer identifier column name as a list
custid = ['customer']

# Select feature column names excluding customer identifier
cols = [col for col in features.columns if col not in custid]

# Extract the features as `X`
X = features[cols]

# Split data to training and testing
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25, random_state=99)

# Initialize linear regression instance
linreg = LinearRegression()

# Fit the model to training dataset
linreg.fit(train_X, train_Y)

# Predict the target variable for training data
train_pred_Y = linreg.predict(train_X)

# Predict the target variable for testing data
test_pred_Y = linreg.predict(test_X)

# Calculate root mean squared error on training data
rmse_train = np.sqrt(mean_squared_error(train_Y, train_pred_Y))

# Calculate mean absolute error on training data
mae_train = mean_absolute_error(train_Y, train_pred_Y)

# Calculate root mean squared error on testing data
rmse_test = np.sqrt(mean_squared_error(test_Y, test_pred_Y))

# Calculate mean absolute error on testing data
mae_test = mean_absolute_error(test_Y, test_pred_Y)

# Print the performance metrics
print('RMSE train: {}; RMSE test: {}\nMAE train: {}, MAE test: {}'.format(rmse_train, rmse_test, mae_train, mae_test))

# Convert train_Y to numpy array
numpy_X = np.asarray(train_X)
numpy_X = numpy_X.astype('float64')
train_X = train_X.astype('float64')
numpy_Y = np.asarray(train_Y)
numpy_Y = numpy_Y.astype('int64')

# Initialize model instance on the training data
olsreg = sm.OLS(numpy_Y, train_X)

# Fit the model
olsreg = olsreg.fit()

# Print model summary
print(olsreg.summary())

# Create new values - pre march
pre_march_df = unnested_df[unnested_df['date_created'] <= '2024-02-29']

# Define the new snapshot date
NOW = dt.datetime(2024, 3, 1).date()

# Calculate recency by subtracting current date from the latest InvoiceDate
new_features = pre_march_df.groupby('billing_email').agg({
  'date_created': lambda x: (NOW - x.max().date()).days,
  # Calculate frequency by counting unique number of invoices
  'order_id': pd.Series.nunique,
  # Calculate monetary value by summing all spend values
  'total': np.sum,
  # Calculate average and total quantity
  'quantity': ['mean', 'sum']}).reset_index()

# Rename the columns
new_features.columns = ['customer', 'recency', 'frequency', 'monetary', 'quantity_avg', 'quantity_total']

# Store customer identifier column name as a list
custid = ['customer']

# Select feature column names excluding customer identifier
cols = [col for col in features.columns if col not in custid]

# Extract the features as `X`
new_X = new_features[cols]

# Make a prediction for march 2024
predicted_value_march_2024 = linreg.predict(new_X)
total_orders_march_2024 = np.sum(predicted_value_march_2024)

# Print the predicted values for march 2024
print("Predicted value for march 2024:", total_orders_march_2024)