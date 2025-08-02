import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

def find_optimal_k(X, max_k=10):
    """Plot elbow curve to find the optimal number of clusters."""
    wcss = []
    for i in range(1, max_k + 1):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
        km.fit(X)
        wcss.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.grid(True)
    plt.savefig("elbow_curve.png")
    plt.show()

def fit_kmeans(X, k=5):
    """Fit KMeans and return model + labels."""
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    y_kmeans = km.fit_predict(X)
    return km, y_kmeans
    
def plot_clusters(X, y_kmeans):
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']
    for i in range(max(y_kmeans) + 1):
        plt.scatter(X[y_kmeans == i, 1], X[y_kmeans == i, 2], 
                    s=100, c=colors[i], label=f'Cluster {i+1}')
    plt.title("Customer Segments (Income vs Spending)")
    plt.xlabel("Annual Income (scaled)")
    plt.ylabel("Spending Score (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("customer_clusters.png")
    plt.show()
