import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    df = df.copy()
    df.drop("CustomerID", axis=1, inplace=True)
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df
