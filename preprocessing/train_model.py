import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from preprocess import preprocess_data
import numpy as np

def train_and_save_model():
    # Load and preprocess the dataset
    file_path = 'data/soil_data.csv'  # Update with the actual dataset path
    data = preprocess_data(file_path)

    # Print the columns for debugging
    print("Columns in DataFrame:", data.columns.tolist())

    # Define features (X) and create a dummy target variable (y)
    columns_to_drop = ['fips', 'lat', 'lon']  # Keep only existing columns
    X = data.drop(columns=columns_to_drop, errors='ignore')

    # Create a dummy target variable
    # For example, use a random binary array as a placeholder
    np.random.seed(42)  # For reproducibility
    y = np.random.randint(0, 2, size=X.shape[0])  # Binary target variable

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)

    # Save the trained model to the model folder
    # joblib.dump(rf_model, 'model/rf_model.pkl')
    print(X_train.columns)
# Run the training function
train_and_save_model()
