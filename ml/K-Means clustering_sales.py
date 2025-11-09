# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
# Make sure 'sales_data_sample.csv' is in the same directory
data = pd.read_csv("sales_data_sample.csv", encoding='latin1')

# Step 2: Display dataset info
print("Dataset Shape:", data.shape)
print(data.head())

# Step 3: Select relevant numerical features for clustering
# (You can adjust columns as needed)
numeric_features = ['QUANTITYORDERED', 'PRICEEACH', 'SALES', 'MSRP']

# Drop rows with missing values in selected columns
df = data[numeric_features].dropna()

# Step 4: Scale the data (important for clustering)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 5: Use the Elbow Method to find the optimal number of clusters
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Step 6: Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Step 7: Apply K-Means with the chosen number of clusters (e.g., k=3)
optimal_k = 3  # Change based on elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 8: Display cluster centers and count
print("\nCluster Centers (in scaled form):")
print(kmeans.cluster_centers_)

print("\nCluster Counts:")
print(df['Cluster'].value_counts())

# Step 9: Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['SALES'], df['PRICEEACH'], c=df['Cluster'], cmap='viridis')
plt.xlabel('SALES')
plt.ylabel('PRICEEACH')
plt.title('K-Means Clustering on Sales Data')
plt.show()
# 🧩 K-MEANS CLUSTERING ON SALES DATA — EXPLANATION 🧩
# -------------------------------------------------------
# This project performs **Customer/Sales Data Segmentation** using the **K-Means Clustering Algorithm**.
#
# The dataset "sales_data_sample.csv" is first loaded and a few important numerical features such as 
# 'QUANTITYORDERED', 'PRICEEACH', 'SALES', and 'MSRP' are selected for clustering analysis.
# These features represent product quantity, unit price, total sale value, and market suggested retail price.
#
# Since clustering depends on distance calculations, the data is standardized using **StandardScaler** 
# to ensure all features have equal weight (mean = 0, standard deviation = 1).
#
# To decide the optimal number of clusters (k), the **Elbow Method** is used — 
# where models are trained for k = 1 to 10, and the corresponding **inertia (SSE)** values are plotted.  
# The “elbow point” in this curve indicates the best number of clusters (usually where SSE starts flattening).
#
# After choosing an optimal k (e.g., 3), the **KMeans algorithm** groups similar sales patterns together 
# based on numerical similarity. Each row in the dataset is assigned a cluster label (0, 1, or 2).
#
# Cluster centers (in scaled form) represent the average feature values for each group, 
# and cluster counts show how many data points belong to each segment.
#
# Finally, the clusters are visualized using a **scatter plot of SALES vs PRICEEACH**, 
# with different colors representing different customer/product groups.  
# This helps in identifying trends like high-price-high-sale products or low-price-mass-sale segments.
