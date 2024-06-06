import pandas_gbq
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

# Create column names list and same length integer list
x_names = product_sales_table.columns
x_ix = np.arange(product_sales_table.shape[1])

# Create averages and standard deviations
averages = product_sales_table.mean()
std_devs = product_sales_table.std()

# Plot the averages data in gray and standard deviations in orange 
plt.bar(x=x_ix-0.2, height=averages, color='grey', label='Average', width=0.4)
plt.bar(x=x_ix+0.2, height=std_devs, color='orange', label='Standard Deviation', width=0.4)

# Add x-axis labels and rotate
plt.xticks(ticks=x_ix, labels=x_names, rotation=90)

# Add the legend and display the chart
plt.legend()

# Save the figure
plt.savefig('product_mean_std.jpeg')

# Close the plot
plt.close()

