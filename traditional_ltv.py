import pandas_gbq
import json
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
monthly_revenue = order_table.groupby(['billing_email','year_month'])['total'].sum().mean()
print(monthly_revenue)

# Define full table id
project_id = os.getenv("COHORT_TABLE_PROJECT_ID")
dataset_id = os.getenv("COHORT_TABLE_DATASET_ID")
table_id = os.getenv("COHORT_TABLE_TABLE_ID")
full_table_id = f"{project_id}.{dataset_id}.{table_id}"

# SQL-query
query = f'SELECT * FROM `{full_table_id}`'

# Use GBQ to get data from BigQuery
cohort_table = pandas_gbq.read_gbq(query, project_id=f'{project_id}')

# Set index
cohort_table.set_index('cohort_month', inplace=True)

# Extract cohort sizes from the first column of cohort_counts
cohort_sizes = cohort_table.iloc[:,0]

# Calculate retention by dividing the counts with the cohort sizes
retention = cohort_table.divide(cohort_sizes, axis=0)

# Calculate churn
churn = 1 - retention

# Calculate the mean retention rate
retention_rate = retention.iloc[:,1:].mean().mean()

# Calculate the mean churn rate
churn_rate = churn.iloc[:,1:].mean().mean()

# Calculate traditional CLV 
clv_traditional = monthly_revenue * (retention_rate / churn_rate)

# Print traditional CLV and the retention rate values
print('Average traditional CLV is {:.1f} EUR at {:.1f} % retention_rate'.format(clv_traditional, retention_rate*100))
