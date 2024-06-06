import pandas_gbq
import os
import numpy as np
from dotenv import load_dotenv

# Load the variables from the .env file
load_dotenv()

# Get the GCP keys
gc_keys = os.getenv("AARDG_GOOGLE_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gc_keys

# Define full table id
unnested_project_id = os.getenv("UNNESTED_ORDER_DATA_PROJECT_ID")
dataset_id = os.getenv("UNNESTED_ORDER_DATA_DATASET_ID")
table_id = os.getenv("UNNESTED_ORDER_DATA_TABLE_ID")
unnested_full_table_id = f"{unnested_project_id}.{dataset_id}.{table_id}"

# SQL-query
unnested_query = f"SELECT *, FORMAT_DATE('%Y-%m', DATE_TRUNC(date_created, MONTH)) AS year_month FROM `{unnested_full_table_id}`"

# Use GBQ to get data from BigQuery
order_table = pandas_gbq.read_gbq(unnested_query, project_id=f'{unnested_project_id}')

# Calculate monthly spend per customer
revenue_per_purchase = order_table.groupby(['order_id'])['total'].mean().mean()
print(revenue_per_purchase)

# Calculate average number of unique invoices per customer per month
frequency_per_month = order_table.groupby(['billing_email','year_month'])['order_id'].nunique().mean()
print(frequency_per_month)

# Define full table id
order_project_id = os.getenv("ORDER_DATA_PROJECT_ID")
dataset_id = os.getenv("ORDER_DATA_DATASET_ID")
table_id = os.getenv("ORDER_DATA_TABLE_ID")
order_full_table_id = f"{order_project_id}.{dataset_id}.{table_id}"

# SQL-query
order_query = f'WITH difference AS (SELECT billing_email, DATE_DIFF(MAX(DATE(date_created)), MIN(DATE(date_created)), MONTH) AS months_difference FROM `{order_full_table_id}` GROUP BY billing_email) SELECT AVG(months_difference) AS lifespan_months FROM difference'

# Use GBQ to get data from BigQuery
lifespan_months = pandas_gbq.read_gbq(order_query, project_id=f'{order_project_id}')

# Extract the average lifespan in months from the dataframe
lifespan_in_months = lifespan_months['lifespan_months'].iloc[0]

# Calculate granular CLV
ltv_granular = revenue_per_purchase * frequency_per_month * lifespan_in_months

# Print granular CLV value
print('Average granular LTV is {:.1f} EUR'.format(ltv_granular))
