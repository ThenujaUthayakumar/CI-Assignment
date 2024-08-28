from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from joblib import load

app = Flask(__name__)

df = pd.read_csv('train.csv')

encoding_maps = {
    'id': {val: idx for idx, val in enumerate(df['id'].unique())},
    'cap-diameter': {val: idx for idx, val in enumerate(df['cap-diameter'].unique())},
    'cap-shape': {val: idx for idx, val in enumerate(df['cap-shape'].unique())},
    'cap-surface': {val: idx for idx, val in enumerate(df['cap-surface'].unique())},
    'cap-color': {val: idx for idx, val in enumerate(df['cap-color'].unique())},
    'does-bruise-or-bleed': {val: idx for idx, val in enumerate(df['does-bruise-or-bleed'].unique())},
    'gill-attachment': {val: idx for idx, val in enumerate(df['gill-attachment'].unique())},
    'gill-spacing': {val: idx for idx, val in enumerate(df['gill-spacing'].unique())},
    'gill-color': {val: idx for idx, val in enumerate(df['gill-color'].unique())},
    'stem-height': {val: idx for idx, val in enumerate(df['stem-height'].unique())},
    'stem-width': {val: idx for idx, val in enumerate(df['stem-width'].unique())},
    'stem-root': {val: idx for idx, val in enumerate(df['stem-root'].unique())},
    'stem-surface': {val: idx for idx, val in enumerate(df['stem-surface'].unique())},
    'stem-color': {val: idx for idx, val in enumerate(df['stem-color'].unique())},
    'veil-type': {val: idx for idx, val in enumerate(df['veil-type'].unique())},
    'veil-color': {val: idx for idx, val in enumerate(df['veil-color'].unique())},
    'has-ring': {val: idx for idx, val in enumerate(df['has-ring'].unique())},
    'ring-type': {val: idx for idx, val in enumerate(df['ring-type'].unique())},
    'spore-print-color': {val: idx for idx, val in enumerate(df['spore-print-color'].unique())},
    'habitat': {val: idx for idx, val in enumerate(df['habitat'].unique())},
    'season': {val: idx for idx, val in enumerate(df['season'].unique())},

}

model = load('mushroom_model_system.joblib')

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    df = pd.DataFrame([data])

    for column in encoding_maps:
        if column in df.columns:
            df[column] = df[column].astype(str).map(encoding_maps[column]).fillna(0).astype(int)
        else:
            df[column] = 0

    all_columns = list(encoding_maps.keys())
    for column in all_columns:
        if column not in df.columns:
            df[column] = 0
    
    df = df[all_columns]

    prediction = model.predict(df)
    
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
