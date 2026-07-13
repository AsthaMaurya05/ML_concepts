from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


dataset = load_wine()
x = dataset.data
y = dataset.target

model = KMeans(n_clusters=3, random_state=42)
model.fit(x)

y_pred = model.labels_

print("K-Means Clustering")
print("Silhouette Score:", silhouette_score(x, y_pred))
print("Adjusted Rand Score:", adjusted_rand_score(y, y_pred))
