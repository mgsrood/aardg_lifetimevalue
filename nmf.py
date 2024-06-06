import pandas_gbq
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import NMF
from dotenv import load_dotenv

# Load the variables from the .env file
load_dotenv()

# Get the GCP keys
gc_keys = os.getenv("AARDG_GOOGLE_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gc_keys

# Define full table id
nmf_project_id = os.getenv("KMEANS_2_PROJECT_ID")
dataset_id = os.getenv("KMEANS_2_DATASET_ID")
table_id = os.getenv("KMEANS_2_TABLE_ID")
nmf_full_table_id = f"{nmf_project_id}.{dataset_id}.{table_id}"

# SQL-query
kmeans_query = f"""SELECT
  billing_email,
  ltv,
  total_order_count,
  avg_order_value,
  avg_time_between_orders,
  non_subscription_orders,
  subscription_orders
FROM `{nmf_full_table_id}`
WHERE ltv != 0"""

# Use GBQ to get data from BigQuery
nmf_table = pandas_gbq.read_gbq(kmeans_query, project_id=f'{nmf_project_id}')

# Set month as index
nmf_table.set_index("billing_email", inplace=True)

# Initialize NMF instance with 4 components
nmf = NMF(4)

# Fit the model on the wholesale sales data
nmf.fit(nmf_table)

# Extract the components 
components = pd.DataFrame(data=nmf.components_, columns=nmf_table.columns)

# Create the W matrix
W = pd.DataFrame(data=nmf.transform(nmf_table), columns=components.index)
W.index = nmf_table.index

# Assign the column name where the corresponding value is the largest
nmf = nmf_table.assign(segment = W.idxmax(axis=1))

# Calculate the average column values per each segment
nmf_averages = nmf.groupby('segment').mean().round(0)

# Convert nmf4_averages to float
nmf_averages = nmf_averages.astype(float)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Plot the average values as heatmap
sns.heatmap(nmf_averages.T, cmap='YlGnBu', fmt='.2f', annot=True, linewidths=.5)

# Adjust the labels for readability
plt.yticks(rotation=0, fontsize=10)  
plt.xticks(rotation=45, fontsize=10)

# Save the figure
plt.savefig('1_heatmap_nmf.jpeg', bbox_inches='tight')

# Close the plot
plt.close()