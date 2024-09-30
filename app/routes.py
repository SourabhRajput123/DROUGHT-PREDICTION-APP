<<<<<<< HEAD
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

from app import app

# Load your trained model
rf_model = joblib.load(r'D:\Projects\drought-prediction-app\model\rf_model.pkl')

# Load your CSV data
data = pd.read_csv(r'D:\Projects\drought-prediction-app\data\soil_data.csv')

@app.route('/')
def index():
    return render_template('index.html')  # Make sure index.html is in the templates folder

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    latitude = float(content['latitude'])
    longitude = float(content['longitude'])
    
    # Find the corresponding feature values based on latitude and longitude
    row = data[(data['lat'] == latitude) & (data['lon'] == longitude)]
    
    if row.empty:
        return jsonify({'error': 'Coordinates not found in the data'}), 404
    
    # Extract and expand feature values to match the model's expected format
    features = row[['elevation', 'slope1', 'slope2', 'slope3', 'slope4', 'slope5', 
                    'slope6', 'slope7', 'slope8', 'aspectN', 'aspectE', 'aspectS', 
                    'aspectW', 'aspectUnknown', 'WAT_LAND', 'NVG_LAND', 'URB_LAND', 
                    'GRS_LAND', 'FOR_LAND', 'CULTRF_LAND', 'CULTIR_LAND', 
                    'CULT_LAND', 'SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']].copy()

    # Expand the SQ columns (example logic, adapt based on how the model expects the SQ features)
    for i in range(1, 8):
        features[f'SQ{i}_1'] = features[f'SQ{i}'] * 0.1  # Example split
        features[f'SQ{i}_2'] = features[f'SQ{i}'] * 0.2  # Example split
        features[f'SQ{i}_3'] = features[f'SQ{i}'] * 0.3  # Example split
        features[f'SQ{i}_4'] = features[f'SQ{i}'] * 0.4  # Example split
        features[f'SQ{i}_6'] = features[f'SQ{i}'] * 0.6  # Example split
        features[f'SQ{i}_7'] = features[f'SQ{i}'] * 0.7  # Example split
        features.drop(columns=[f'SQ{i}'], inplace=True)   # Remove the original SQ column

    # Create DataFrame for the model
    features_df = pd.DataFrame(features.values.reshape(1, -1), columns=[
        'elevation', 'slope1', 'slope2', 'slope3', 'slope4', 'slope5', 
        'slope6', 'slope7', 'slope8', 'aspectN', 'aspectE', 'aspectS', 
        'aspectW', 'aspectUnknown', 'WAT_LAND', 'NVG_LAND', 'URB_LAND', 
        'GRS_LAND', 'FOR_LAND', 'CULTRF_LAND', 'CULTIR_LAND', 'CULT_LAND',
        'SQ1_1', 'SQ1_2', 'SQ1_3', 'SQ1_4', 'SQ1_6', 'SQ1_7',  
        'SQ2_1', 'SQ2_2', 'SQ2_3', 'SQ2_4', 'SQ2_6', 'SQ2_7',  
        'SQ3_1', 'SQ3_2', 'SQ3_3', 'SQ3_4', 'SQ3_6', 'SQ3_7',  
        'SQ4_1', 'SQ4_2', 'SQ4_3', 'SQ4_4', 'SQ4_6', 'SQ4_7',  
        'SQ5_1', 'SQ5_2', 'SQ5_3', 'SQ5_4', 'SQ5_6', 'SQ5_7',  
        'SQ6_1', 'SQ6_2', 'SQ6_3', 'SQ6_6', 'SQ6_7',            
        'SQ7_1', 'SQ7_2', 'SQ7_3', 'SQ7_4', 'SQ7_5', 'SQ7_6', 'SQ7_7'
    ])
    
    # Make prediction
    drought_prediction = rf_model.predict(features_df)

    # Convert the result to Python int and return JSON response
    return jsonify({'drought_prediction': int(drought_prediction[0])})

@app.route('/test')
def test():
    return "This is a test page"

if __name__ == '__main__':
    app.run(debug=True)
=======
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

from app import app

# Load your trained model
rf_model = joblib.load(r'D:\Projects\drought-prediction-app\model\rf_model.pkl')

# Load your CSV data
data = pd.read_csv(r'D:\Projects\drought-prediction-app\data\soil_data.csv')

@app.route('/')
def index():
    return render_template('index.html')  # Make sure index.html is in the templates folder

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    latitude = float(content['latitude'])
    longitude = float(content['longitude'])
    
    # Find the corresponding feature values based on latitude and longitude
    row = data[(data['lat'] == latitude) & (data['lon'] == longitude)]
    
    if row.empty:
        return jsonify({'error': 'Coordinates not found in the data'}), 404
    
    # Extract and expand feature values to match the model's expected format
    features = row[['elevation', 'slope1', 'slope2', 'slope3', 'slope4', 'slope5', 
                    'slope6', 'slope7', 'slope8', 'aspectN', 'aspectE', 'aspectS', 
                    'aspectW', 'aspectUnknown', 'WAT_LAND', 'NVG_LAND', 'URB_LAND', 
                    'GRS_LAND', 'FOR_LAND', 'CULTRF_LAND', 'CULTIR_LAND', 
                    'CULT_LAND', 'SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7']].copy()

    # Expand the SQ columns (example logic, adapt based on how the model expects the SQ features)
    for i in range(1, 8):
        features[f'SQ{i}_1'] = features[f'SQ{i}'] * 0.1  # Example split
        features[f'SQ{i}_2'] = features[f'SQ{i}'] * 0.2  # Example split
        features[f'SQ{i}_3'] = features[f'SQ{i}'] * 0.3  # Example split
        features[f'SQ{i}_4'] = features[f'SQ{i}'] * 0.4  # Example split
        features[f'SQ{i}_6'] = features[f'SQ{i}'] * 0.6  # Example split
        features[f'SQ{i}_7'] = features[f'SQ{i}'] * 0.7  # Example split
        features.drop(columns=[f'SQ{i}'], inplace=True)   # Remove the original SQ column

    # Create DataFrame for the model
    features_df = pd.DataFrame(features.values.reshape(1, -1), columns=[
        'elevation', 'slope1', 'slope2', 'slope3', 'slope4', 'slope5', 
        'slope6', 'slope7', 'slope8', 'aspectN', 'aspectE', 'aspectS', 
        'aspectW', 'aspectUnknown', 'WAT_LAND', 'NVG_LAND', 'URB_LAND', 
        'GRS_LAND', 'FOR_LAND', 'CULTRF_LAND', 'CULTIR_LAND', 'CULT_LAND',
        'SQ1_1', 'SQ1_2', 'SQ1_3', 'SQ1_4', 'SQ1_6', 'SQ1_7',  
        'SQ2_1', 'SQ2_2', 'SQ2_3', 'SQ2_4', 'SQ2_6', 'SQ2_7',  
        'SQ3_1', 'SQ3_2', 'SQ3_3', 'SQ3_4', 'SQ3_6', 'SQ3_7',  
        'SQ4_1', 'SQ4_2', 'SQ4_3', 'SQ4_4', 'SQ4_6', 'SQ4_7',  
        'SQ5_1', 'SQ5_2', 'SQ5_3', 'SQ5_4', 'SQ5_6', 'SQ5_7',  
        'SQ6_1', 'SQ6_2', 'SQ6_3', 'SQ6_6', 'SQ6_7',            
        'SQ7_1', 'SQ7_2', 'SQ7_3', 'SQ7_4', 'SQ7_5', 'SQ7_6', 'SQ7_7'
    ])
    
    # Make prediction
    drought_prediction = rf_model.predict(features_df)

    # Convert the result to Python int and return JSON response
    return jsonify({'drought_prediction': int(drought_prediction[0])})

@app.route('/test')
def test():
    return "This is a test page"

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> ff399a1 (Compleated Project)
