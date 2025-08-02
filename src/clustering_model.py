from sklearn.cluster import KMeans

def fit_kmeans(X, k=5):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    y_kmeans = km.fit_predict(X)
    return km, y_kmeans
