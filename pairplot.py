import pandas_gbq
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
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

'''# Apply a box cox transformation
def boxcox_df(x):
    x_positive = x - x.min() + 1
    x_boxcox, _ = stats.boxcox(x_positive)
    return x_boxcox

product_sales_table_boxcox = product_sales_table.apply(boxcox_df, axis=0)

# Apply a log transformation
product_sales_table_log = np.log(product_sales_table)'''

# Plot the pairwise relationships between the variables
sns.pairplot(product_sales_table, diag_kind='kde')

# Save the figure
plt.savefig('pairplot.jpeg')

# Close the plot
plt.close()

