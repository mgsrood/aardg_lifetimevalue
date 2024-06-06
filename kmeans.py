import pandas_gbq
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
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
unnested_query = f"""WITH base AS (
  SELECT *
  FROM (
    SELECT 
      billing_email,
      base_product,
      COALESCE(quantity, 0) AS quantity
    FROM `{unnested_full_table_id}`
  )
  PIVOT (
    SUM(COALESCE(quantity, 0))
    FOR base_product IN ('bloem_kombucha', 'mix_originals_2', 'starter_box', 'citroen_kombucha', 'kombucha_original', 'probiotica_ampullen', 'mix_originals', 'waterkefir_original', 'frisdrank_mix', 'gember_limonade')
  )
)
SELECT
  billing_email,
  IFNULL(bloem_kombucha, 0) AS bloem_kombucha,
  IFNULL(mix_originals_2, 0) AS mix_originals_2,
  IFNULL(starter_box, 0) AS starter_box,
  IFNULL(citroen_kombucha, 0) AS citroen_kombucha,
  IFNULL(kombucha_original, 0) AS kombucha_original,
  IFNULL(probiotica_ampullen, 0) AS probiotica_ampullen,
  IFNULL(mix_originals, 0) AS mix_originals,
  IFNULL(waterkefir_original, 0) AS waterkefir_original,
  IFNULL(frisdrank_mix, 0) AS frisdrank_mix,
  IFNULL(gember_limonade, 0) AS gember_limonade
FROM base"""

# Use GBQ to get data from BigQuery
product_sales_table = pandas_gbq.read_gbq(unnested_query, project_id=f'{unnested_project_id}')

# Set month as index
product_sales_table.set_index("billing_email", inplace=True)

# Set up scaling
scaler = StandardScaler()
scaler.fit(product_sales_table)
scaled_table = scaler.transform(product_sales_table)
scaled_table_df = pd.DataFrame(scaled_table, index=product_sales_table.index, columns=product_sales_table.columns)

'''# Determine the optimal number of clusters
sse = {}

# Fit KMeans algorithm on k values between 1 and 11
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=333)
    kmeans.fit(product_sales_table)
    sse[k] = kmeans.inertia_

# Add the title to the plot
plt.title('Elbow criterion method chart')

# Create and display a scatter plot
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))

# Save the figure
plt.savefig('elbow_method.jpeg')

# Close the plot
plt.close()'''

# Initiate KMeans 
kmeans=KMeans(n_clusters=4, random_state=123)

# Fit the model on the pre-processed dataset
kmeans.fit(scaled_table_df)

# Assign the generated labels to a new column
product_sales_table_kmeans4 = product_sales_table.assign(segment = kmeans.labels_)

# Group by the segment label and calculate average column values
kmeans4_averages = product_sales_table_kmeans4.groupby(['segment']).mean().round(0)

# Switch to numeric
kmeans4_averages = kmeans4_averages.astype(float)

# Create a heatmap on the average column values per each segment
sns.heatmap(kmeans4_averages.T, cmap='YlGnBu')

# Save the figure
plt.savefig('heatmap_kmeans.jpeg')

# Close the plot
plt.close()