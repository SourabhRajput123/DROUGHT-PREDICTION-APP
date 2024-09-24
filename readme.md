# Drought Prediction Application

## Introduction

The Drought Prediction Application is a web-based tool designed to predict drought conditions based on geographical coordinates (latitude and longitude). This application leverages machine learning algorithms to analyze environmental data and provide predictions, enabling better decision-making for agricultural planning and water resource management.

## Features

- User-friendly interface for inputting geographical coordinates.
- Machine learning model to predict drought conditions.
- Real-time prediction results displayed on the same page.

## Dependencies

This application requires the following dependencies:

- Python 3.11.3
- Flask
- Pandas
- Scikit-learn
- Any other required libraries specified in `requirements.txt`

## File Structure

```
drought-prediction-app/
│
├── app/
│   ├── __init__.py         # Initializes the Flask application
│   ├── routes.py           # Contains the application routes
│   ├── model.py            # Contains the machine learning model and prediction logic
│   └── templates/          # Directory for HTML templates
│       └── index.html      # Main HTML file for the application
│
├── data/
│   ├── soil_data.csv       # CSV file with environmental data
│
├── venv/                    # Virtual environment directory
│
├── requirements.txt         # List of dependencies
│
└── README.md                # This README file
```

## Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/drought-prediction-app.git
   cd drought-prediction-app
   ```

2. **Set Up a Virtual Environment**
   - If you're using `venv`, you can create one with:
   ```bash
   python -m venv venv
   ```
   - Activate the virtual environment:
     - On Windows:
     ```bash
     venv\Scripts\activate
     ```
     - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   flask run
   ```
   - Open your browser and go to `http://127.0.0.1:5000` to access the application.

## Usage

- Enter the latitude and longitude in the input fields provided on the main page.
- Click the "Predict" button to obtain the drought prediction result.
- The prediction result will be displayed directly on the page.

## Contributing

Contributions are welcome! If you have suggestions for improvements or features, please create an issue or submit a pull request.


## Acknowledgments

- Thanks to the open-source community for the libraries and resources that made this project possible.
