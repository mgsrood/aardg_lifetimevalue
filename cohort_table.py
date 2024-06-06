import pandas_gbq
import os
from dotenv import load_dotenv

# Load the variables from the .env file
load_dotenv()

# Get the GCP keys
gc_keys = os.getenv("AARDG_GOOGLE_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gc_keys

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

# Define the years
years = ['2018', '2019', '2020', '2021', '2022', '2023']

# Loop over the years
for year in years:

    # Make a specific selection
    cohort_table_selected = cohort_table[cohort_table.index.str.startswith(year)]

    # Extract cohort sizes from the first column of cohort_counts
    cohort_sizes = cohort_table_selected.iloc[:,0]

    # Calculate retention by dividing the counts with the cohort sizes
    retention = cohort_table_selected.divide(cohort_sizes, axis=0)

    # Calculate churn
    churn = 1 - retention

    # Calculate the mean retention rate
    retention_rate = retention.iloc[:,1:].mean().mean()

    # Calculate the mean churn rate
    churn_rate = churn.iloc[:,1:].mean().mean()

    # Print rounded retention and churn rates
    print('{} Retention Rate: {:.2f}; Churn Rate: {:.2f}'.format(year, retention_rate, churn_rate))

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

# Print rounded retention and churn rates
print('Overall Retention Rate: {:.2f}; Churn Rate: {:.2f}'.format(retention_rate, churn_rate))

