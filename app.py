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

# Generate synthetic datasets for different MBA scenarios
def generate_marketing_data(n_samples=300):
    np.random.seed(42)
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
            'ID': f"CUST-{i + 1}",
            'Feature1Name': 'Annual Income ($k)',
            'Feature2Name': 'Spending Score',
            'Feature1': int(np.random.normal((params['income'][0] + params['income'][1])/2, 10)),
            'Feature2': int(np.random.normal((params['spending'][0] + params['spending'][1])/2, 10))
        })
    df = pd.DataFrame(data)
    df['Feature1'] = df['Feature1'].clip(10, 140)
    df['Feature2'] = df['Feature2'].clip(1, 100)
    return df

def generate_hr_data(n_samples=300):
    np.random.seed(42)
    data = []
    # Clusters: 
    # 1. Burned out (high hours, low sat)
    # 2. Happy/Core (med hours, high sat)
    # 3. Underutilized (low hours, med sat)
    clusters_params = [
        {'hours': (240, 300), 'sat': (10, 30)},
        {'hours': (140, 180), 'sat': (70, 95)},
        {'hours': (90, 130), 'sat': (40, 60)}
    ]
    for i in range(n_samples):
        params = random.choice(clusters_params)
        data.append({
            'ID': f"EMP-{i + 1}",
            'Feature1Name': 'Monthly Hours',
            'Feature2Name': 'Satisfaction (1-100)',
            'Feature1': int(np.random.normal((params['hours'][0] + params['hours'][1])/2, 15)),
            'Feature2': int(np.random.normal((params['sat'][0] + params['sat'][1])/2, 8))
        })
    df = pd.DataFrame(data)
    df['Feature1'] = df['Feature1'].clip(80, 320)
    df['Feature2'] = df['Feature2'].clip(1, 100)
    return df

def generate_product_data(n_samples=300):
    np.random.seed(42)
    data = []
    # Clusters: 
    # 1. Luxury (High Price, Low Vol)
    # 2. Mass Market (Low Price, High Vol)
    # 3. Mid-Tier (Med Price, Med Vol)
    # 4. Cash Cows (Med/High Price, High Vol)
    clusters_params = [
        {'price': (200, 300), 'vol': (10, 50)},
        {'price': (10, 40), 'vol': (400, 600)},
        {'price': (70, 120), 'vol': (150, 250)},
        {'price': (140, 190), 'vol': (300, 500)}
    ]
    for i in range(n_samples):
        params = random.choice(clusters_params)
        data.append({
            'ID': f"SKU-{i + 1}",
            'Feature1Name': 'Price ($)',
            'Feature2Name': 'Monthly Units Sold',
            'Feature1': int(np.random.normal((params['price'][0] + params['price'][1])/2, 20)),
            'Feature2': int(np.random.normal((params['vol'][0] + params['vol'][1])/2, 40))
        })
    df = pd.DataFrame(data)
    df['Feature1'] = df['Feature1'].clip(5, 350)
    df['Feature2'] = df['Feature2'].clip(5, 700)
    return df

datasets = {
    'marketing': generate_marketing_data(),
    'hr': generate_hr_data(),
    'product': generate_product_data()
}

@app.route('/dataset', methods=['GET'])
def get_dataset():
    scenario = request.args.get('scenario', 'marketing')
    df = datasets.get(scenario, datasets['marketing'])
    return jsonify(df.to_dict(orient='records'))

@app.route('/cluster', methods=['POST'])
def cluster_data():
    content = request.json
    k = content.get('k', 3)
    scenario = content.get('scenario', 'marketing')
    df = datasets.get(scenario, datasets['marketing'])
    
    X = df[['Feature1', 'Feature2']].values
    
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
    scenario = request.args.get('scenario', 'marketing')
    df = datasets.get(scenario, datasets['marketing'])
    X = df[['Feature1', 'Feature2']].values
    
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(float(kmeans.inertia_))
    
    return jsonify({'wcss': wcss})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
