import joblib
import pandas as pd
import re
import streamlit as st

# Load the trained models and label encoder
rf_model_outcome = joblib.load("SVC_outcome_best.pkl")
rf_model_disease = joblib.load("RandomForestClassifier_disease_best.pkl")
le_disease = joblib.load("le_Disease.pkl")  # Assuming LabelEncoder is saved as well
# Sample data for X_train columns (replace with actual data structure)
# Ensure this list matches the exact features used during model training
X_train_columns = [
    'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
    'Age', 'Blood Pressure', 'Cholesterol Level'  # Include all features used during training
]

# Define a function to extract relevant features from the input text
def extract_features_from_text(user_input):
    # Define the feature keywords and corresponding Yes/No mappings
    feature_keywords = {
        "fever": "Fever",
        "cough": "Cough",
        "fatigue": "Fatigue",
        "difficulty breathing": "Difficulty Breathing",
        "age": "Age",
        "blood pressure": "Blood Pressure",
        "cholesterol level": "Cholesterol Level"# Include additional features here
    }

    # Initialize a dictionary to store the extracted features
    extracted_features = {
        "Fever": 0,
        "Cough": 0,
        "Fatigue": 0,
        "Difficulty Breathing": 0,
        "Age": 0,
        "Blood Pressure": "Normal",  # Default value, can be updated to 'High' or 'Low'
        "Cholesterol Level": "Normal"  # Default value, can be updated to 'High' or 'Low'

    }

    # Use regular expressions to extract Yes/No, High/Low, and numerical features
    for key, value in feature_keywords.items():
        if value in ["Blood Pressure", "Cholesterol Level"]:
            # Check for High/Low in the user input
            if re.search(r"\bhigh\b", user_input, re.IGNORECASE):
                extracted_features[value] = "High"
            elif re.search(r"\blow\b", user_input, re.IGNORECASE):
                extracted_features[value] = "Low"
            else:
                extracted_features[value] = "Normal"  # Default to normal if not specified
        elif value == "Age":
            # Extract numerical value for Age
            age_match = re.search(r'\bage\s+(\d+)\b', user_input, re.IGNORECASE)
            if age_match:
                extracted_features[value] = int(age_match.group(1))
            else:
                extracted_features[value] = 30  # Default value if age not specified
        else:
            # Check for Yes/No in the user input for other features
            if re.search(rf"\b{key}\b.*\byes\b", user_input, re.IGNORECASE):
                extracted_features[value] = 1  # Yes
            elif re.search(rf"\b{key}\b.*\bno\b", user_input, re.IGNORECASE):
                extracted_features[value] = 0  # No

    return extracted_features

# Define a function to perform inference based on extracted features
def perform_inference_from_text(user_input):
    # Extract features from user input text
    features = extract_features_from_text(user_input)

    # Convert blood pressure and cholesterol levels to numerical encoding
    bp_mapping = {"Low": 0, "Normal": 1, "High": 2}
    chol_mapping = {"Low": 0, "Normal": 1, "High": 2}

    features['Blood Pressure'] = bp_mapping[features['Blood Pressure']]
    features['Cholesterol Level'] = chol_mapping[features['Cholesterol Level']]

    # Convert extracted features to a DataFrame
    input_df = pd.DataFrame([features])

    # Ensure all required columns are present by checking against the original feature set
    missing_cols = set(X_train_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Assign default value for missing columns

    # Ensure the order of columns matches the training data
    input_df = input_df[X_train_columns]

    # Perform inference
    outcome_prediction = rf_model_outcome.predict(input_df)[0]
    disease_prediction = rf_model_disease.predict(input_df)[0]
    predicted_disease_name = le_disease.inverse_transform([disease_prediction])[0]

    # Generate result message
    result_message = (
        f"Predicted Outcome: {'Positive' if outcome_prediction == 1 else 'Negative'}\n"
        f"Predicted Disease: {predicted_disease_name}"
    )
    return result_message

# Initialize the Streamlit app
st.title("Disease and Outcome Prediction")
st.write("Enter your symptoms and other health indicators below:")

# Create input fields in the Streamlit app
user_input = st.text_area("Describe your symptoms and health indicators in natural language:", "")

# Perform prediction when the button is clicked
if st.button("Predict"):
    if user_input:
        result = perform_inference_from_text(user_input)
        st.success(result)
    else:
        st.warning("Please enter your symptoms and health indicators.")
