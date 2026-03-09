from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import random

import os

app = Flask(__name__, 
            static_folder='../frontend', 
            static_url_path='')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Generate synthetic dataset
def generate_data(n_samples=300):
    np.random.seed(42)
    
    # We want to create some natural clusters for MBA students to find
    # Cluster 1: Young, High Income, High Spending
    # Cluster 2: Old, Low Income, Low Spending
    # Cluster 3: Middle aged, Medium Income, Medium Spending
    # Cluster 4: Young, Low Income, High Spending
    # Cluster 5: Old, High Income, Low Spending
    
    data = []
    clusters_params = [
        {'age': (20, 35), 'income': (70, 110), 'spending': (70, 95)},
        {'age': (45, 70), 'income': (15, 40), 'spending': (10, 40)},
        {'age': (30, 50), 'income': (40, 70), 'spending': (40, 60)},
        {'age': (18, 30), 'income': (15, 35), 'spending': (60, 90)},
        {'age': (35, 65), 'income': (75, 120), 'spending': (5, 35)}
    ]
    
    for i in range(n_samples):
        params = random.choice(clusters_params)
        data.append({
            'CustomerID': i + 1,
            'Age': int(np.random.normal((params['age'][0] + params['age'][1])/2, 5)),
            'Annual Income (k$)': int(np.random.normal((params['income'][0] + params['income'][1])/2, 10)),
            'Spending Score (1-100)': int(np.random.normal((params['spending'][0] + params['spending'][1])/2, 10))
        })
    
    df = pd.DataFrame(data)
    # Clip values to realistic ranges
    df['Age'] = df['Age'].clip(18, 75)
    df['Annual Income (k$)'] = df['Annual Income (k$)'].clip(10, 140)
    df['Spending Score (1-100)'] = df['Spending Score (1-100)'].clip(1, 100)
    
    return df

df_customers = generate_data()

@app.route('/dataset', methods=['GET'])
def get_dataset():
    return jsonify(df_customers.to_dict(orient='records'))

@app.route('/cluster', methods=['POST'])
def cluster_data():
    content = request.json
    k = content.get('k', 3)
    
    # We'll use Annual Income and Spending Score for 2D visualization
    X = df_customers[['Annual Income (k$)', 'Spending Score (1-100)']].values
    
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    return jsonify({
        'labels': labels.tolist(),
        'centers': centers.tolist(),
        'inertia': float(inertia)
    })

@app.route('/elbow', methods=['GET'])
def get_elbow():
    X = df_customers[['Annual Income (k$)', 'Spending Score (1-100)']].values
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(float(kmeans.inertia_))
    
    return jsonify({'wcss': wcss})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
