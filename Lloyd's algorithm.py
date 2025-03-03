import pandas as pd
from pprint import pprint
import matplotlib.pyplot as mp
import matplotlib.pyplot as plt
import seaborn as sb 
import math
import numpy as np
from sklearn.decomposition import PCA


df=pd.read_csv('Dry_Bean.csv')
df.head(10)

#Task1

data_state=df.describe()
data_state.loc['sum']=df.sum()
print(data_state)

# Task 2
correlation_matrix = df.drop(columns=['Class']).corr()
highly_correlated_features=correlation_matrix[correlation_matrix>0.8]
highly_correlated_features=highly_correlated_features.dropna(how='all',axis=1).dropna(how='all',axis=0)
selected_features=pd.DataFrame(highly_correlated_features.stack(),columns=['Correlation Coefficient'])
print(selected_features)

#Task3
sb.heatmap(correlation_matrix)
plt.title('Correlation Matrix Heatmap')
plt.show()


#Task 4

# Set the random seed for reproducibility
np.random.seed(42)

# Read in the dataset and perform one-hot encoding
df = pd.read_csv("Dry_Bean.csv")
onehot_encoded_df = pd.get_dummies(df)


# Use PCA to reduce the dimensionality of the dataset to 2 features
pca_model = pd.DataFrame(PCA(n_components=2).fit_transform(onehot_encoded_df))



def kmeans(X, k, epsilon=10**-5):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    fig, ax = plt.subplots()
    counter = 0
    prev_clusters = np.zeros(X.shape[0])


    while True:
        # Assign each data point to the nearest centroid
        distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
        clusters = np.argmin(distances, axis=1)

        # Update the position of the centroids
        new_centroids = np.array([X[clusters == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.max(np.abs(new_centroids - centroids)) < epsilon:
            break
        if np.array_equal(prev_clusters, clusters):
            counter += 1
            if counter == 5:
                break
        else:
            prev_clusters = clusters
            counter = 0

        centroids = new_centroids
        ax.clear()
        ax.scatter(X[:, 0], X[:, 1], c=clusters)
        ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=30, edgecolors='black', linewidths=2, c='black')
        plt.pause(0.1)
    plt.show()


# Call the k-means clustering function 
kmeans(pca_model.values, k=5)