import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Step 1: Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Step 2: Exploratory Data Analysis
print(df.head())
print(df.describe())
print(df.info())

# Visualizing the distribution of the features
plt.figure(figsize=(10, 6))
sns.histplot(df['Annual Income (k$)'], bins=20, kde=True)
plt.title('Distribution of Annual Income')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True)
plt.title('Distribution of Spending Score')
plt.show()

# Step 3: Data Preprocessing
# We'll cluster based on 'Annual Income (k$)' and 'Spending Score (1-100)'
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: K-Means Clustering
# Finding the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Based on the elbow curve, let's choose k=5
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Step 5: Adding the cluster labels to the dataset
df['Cluster'] = kmeans.labels_

# Step 6: Visualizing the Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette='Set1', s=100)
plt.title('Customer Segments Based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# 3D Visualization using plotly for better understanding
fig = px.scatter_3d(df, x='Annual Income (k$)', y='Spending Score (1-100)', z='Age', color='Cluster', size_max=18, opacity=0.7)
fig.update_layout(title='3D Visualization of Customer Segments', scene=dict(xaxis_title='Annual Income (k$)', yaxis_title='Spending Score', zaxis_title='Age'))
fig.show()

# Step 7: Analyzing the Clusters
cluster_analysis = df.groupby('Cluster').mean()
print(cluster_analysis)
