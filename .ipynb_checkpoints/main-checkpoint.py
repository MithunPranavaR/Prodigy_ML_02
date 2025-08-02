from src.data_utils import load_data
from src.preprocessing import preprocess_data
from src.clustering_model import fit_kmeans
from src.visualization import find_optimal_k, plot_clusters
import pandas as pd

if __name__ == "__main__":
    # 1. Load
    df = load_data("data/Mall_Customers.csv")

    # 2. Preprocess
    X_scaled, df_processed = preprocess_data(df)

    # 3. Find optimal K
    find_optimal_k(X_scaled)

    # 4. Train model
    model, labels = fit_kmeans(X_scaled, k=5)

    # 5. Save clustered data
    df_processed['Cluster'] = labels
    df_processed.to_csv("clustered_customers.csv", index=False)

    # 6. Visualize clusters
    plot_clusters(X_scaled, labels)
