# -*- coding: utf-8 -*-
"""Hackathon_Disease_Symptoms.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tIQofztDKqTtkJI87TOyQmFJzkcrI6Pf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#Overall project is to train all the models including neural network and see which model performs the best. And then we use the same model to train a chatbot of LLM to allow the user to prompt. And also future possibility.

# prompt: read the csv file using pandas

import pandas as pd
df = pd.read_csv('/content/Disease_symptom_and_patient_profile_dataset.csv') # Replace 'your_file.csv' with the actual filename

df.head()

df.shape

df.info()

print(df['Disease'].unique())  # List of unique diseases
print(df['Outcome Variable'].value_counts())  # Check distribution of outcomes

plt.figure(figsize=(20, 10))  # Adjust size as needed
sns.countplot(data=df, x='Disease', hue='Outcome Variable')
plt.title('Distribution of Positive and Negative Outcomes for Each Disease')
plt.xticks(rotation=90)  # Rotate x labels for better readability
plt.show()

# Top N diseases based on total cases
top_n = 20  # Adjust this number based on your preference

# Calculate total count for each disease
top_diseases = df['Disease'].value_counts().nlargest(top_n).index

# Filter data for only top N diseases
filtered_data = df[df['Disease'].isin(top_diseases)]

# Plot for top diseases
plt.figure(figsize=(15, 8))
sns.countplot(data=filtered_data, x='Disease', hue='Outcome Variable')
plt.title(f'Distribution of Positive and Negative Outcomes for Top {top_n} Diseases')
plt.xticks(rotation=45)  # Rotate x labels for better readability
plt.show()

# Aggregated count of positive and negative outcomes across all diseases
outcome_counts = df.groupby('Outcome Variable')['Disease'].count().reset_index()

# Bar plot for aggregated counts
plt.figure(figsize=(8, 5))
sns.barplot(data=outcome_counts, x='Outcome Variable', y='Disease')
plt.title('Total Number of Positive and Negative Outcomes Across All Diseases')
plt.ylabel('Count of Diseases')
plt.show()

# Create pivot table without 'values' parameter
heatmap_data = pd.pivot_table(df, index='Disease', columns='Outcome Variable', aggfunc='size', fill_value=0)

# Filter for top N diseases
top_n = 20  # Adjust as needed
top_diseases = df['Disease'].value_counts().nlargest(top_n).index
filtered_heatmap_data = heatmap_data.loc[top_diseases]

# Plot heatmap with filtered data
plt.figure(figsize=(12, 15))
sns.heatmap(filtered_heatmap_data, annot=True, fmt="d", cmap='Blues')
plt.title(f'Heatmap of Outcomes for Top {top_n} Diseases')
plt.xlabel('Outcome')
plt.ylabel('Disease')
plt.show()

#check for missing values

missing_values = df.isnull().sum()
print(missing_values)

# Convert the Outcome Variable to numeric
df['Outcome Numeric'] = df['Outcome Variable'].map({'Positive': 1, 'Negative': 0})

# Check the conversion
print(df[['Outcome Variable', 'Outcome Numeric']].head())

# Calculate the Pearson correlation coefficient between two variables
correlation = df['Age'].corr(df['Outcome Numeric'])
print(f'Pearson correlation coefficient between Age and Outcome Numeric: {correlation:.2f}')

import pandas as pd

# Binary variables to categorical
binary_vars = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
for var in binary_vars:
    df[var] = pd.Categorical(df[var], categories=['No', 'Yes'], ordered=True)

# Categorical variables with levels
categorical_vars = {
    'Blood Pressure': ['Low', 'Normal', 'High'],
    'Cholesterol Level': ['Low', 'Normal', 'High'],
    'Gender': ['Male', 'Female', 'Other']  # Adjust as per your data
}
for var, levels in categorical_vars.items():
    df[var] = pd.Categorical(df[var], categories=levels, ordered=False)

# Check the updated data types and first few rows
print(df.dtypes)
print(df.head())

# If you need to encode categorical variables for machine learning:
# One-hot encoding (if needed for linear models)
# Drop first category to avoid multicollinearity
encoded_df = pd.get_dummies(df, columns=binary_vars + list(categorical_vars.keys()), drop_first=True)

# Display the first few rows of the encoded DataFrame
print(encoded_df.head())

#Chi square contingency
import pandas as pd
from scipy.stats import chi2_contingency

# Assuming your data is in the DataFrame `df`
# Ensure 'Outcome Variable' is binary (0/1 or Yes/No)
# If not, convert it to 0/1
# df['Outcome Variable'] = df['Outcome Variable'].map({'No': 0, 'Yes': 1})  # Uncomment if needed

# List of categorical variables to test
categorical_vars = binary_vars + list(categorical_vars.keys())

# Dictionary to store p-values
chi2_p_values = {}

# Perform Chi-squared test for each categorical variable
for var in categorical_vars:
    # Create contingency table
    contingency_table = pd.crosstab(df[var], df['Outcome Variable'])

    # Perform Chi-squared test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Store the p-value in the dictionary
    chi2_p_values[var] = p

# Convert p-values dictionary to a DataFrame and sort by p-value
p_values_df = pd.DataFrame(list(chi2_p_values.items()), columns=['Predictor', 'P-Value'])

# Format p-values to display in decimal form with 4 decimal places
p_values_df['P-Value'] = p_values_df['P-Value'].map(lambda x: f'{x:.4f}')

# Sort the DataFrame by P-Value
p_values_df = p_values_df.sort_values(by='P-Value')

# Display the sorted p-values
print(p_values_df)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt

# Function to calculate Cramer's V for two categorical variables
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * k))

# Function to create Cramer's V matrix for all pairs of categorical variables
def cramers_v_matrix(df, categorical_columns):
    n = len(categorical_columns)
    cramer_v_matrix = pd.DataFrame(np.zeros((n, n)), index=categorical_columns, columns=categorical_columns)
    for col1 in range(n):
        for col2 in range(col1, n):
            cramer_v_matrix.iloc[col1, col2] = cramers_v(df[categorical_columns[col1]], df[categorical_columns[col2]])
            cramer_v_matrix.iloc[col2, col1] = cramer_v_matrix.iloc[col1, col2]
    return cramer_v_matrix

# List of categorical variables to analyze
categorical_vars = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Blood Pressure', 'Cholesterol Level', 'Gender']

# Calculate Cramer's V matrix
cramer_v_df = cramers_v_matrix(df, categorical_vars)

# Convert Cramer's V to a distance matrix (1 - Cramer's V)
distance_matrix = 1 - cramer_v_df

# Perform hierarchical clustering using the distance matrix
Z = linkage(distance_matrix, method='ward')

# Visualize the dendrogram for variable clustering
plt.figure(figsize=(12, 8))
dendrogram(Z, labels=distance_matrix.index, leaf_rotation=90, leaf_font_size=12)
plt.title("Hierarchical Clustering Dendrogram (Cramer's V Distance)")
plt.xlabel("Categorical Variables")
plt.ylabel("Distance (1 - Cramer's V)")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Plot heatmap for Cramer's V matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cramer_v_df, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title("Cramer's V Matrix for Categorical Variables")
plt.show()

# prompt: drop gender from dataframe variable

# Assuming 'df' is your DataFrame
df = df.drop('Gender', axis=1)

df.info()

df.head()

# prompt: include datatypes from df of category and object as categorical columns

categorical_columns = df.select_dtypes(include=['category', 'object']).columns.tolist()
print(categorical_columns)

import joblib
from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    # Save the encoder for each column with a unique name
    joblib.dump(le, f"le_{col}.pkl")

# Display the DataFrame after label encoding
print("DataFrame after label encoding:")
print(df)

X = df.drop(columns=['Outcome Variable'])

df.info()

# prompt: drop outcome variable in df

df = df.drop('Outcome Variable', axis=1)

df.head(15)

X.info()

# prompt: install library for bayes optimization

!pip install bayesian-optimization

import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np

# Assuming df is your DataFrame with data loaded

# Step 1: Prepare your data
X = df.drop(columns=['Outcome Numeric','Disease'])  # Drop both target columns from predictors
y_outcome = df['Outcome Numeric']  # Target variable for outcome
y_disease = df['Disease']  # Target variable for disease

# Step 2: Split the data into training, validation, and test sets for both outcomes
X_train_full, X_test, y_outcome_train_full, y_outcome_test, y_disease_train_full, y_disease_test = train_test_split(
    X, y_outcome, y_disease, test_size=0.3, stratify=y_outcome, random_state=42
)
X_train, X_val, y_outcome_train, y_outcome_val, y_disease_train, y_disease_val = train_test_split(
    X_train_full, y_outcome_train_full, y_disease_train_full, test_size=0.125, stratify=y_outcome_train_full, random_state=42
)

# Display split data sizes
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Step 3: Define Cross-Validation Strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Step 4: Define the Objective Functions for Bayesian Optimization for Different Models

# Random Forest Classifier for Outcome
def rf_cv_outcome(n_estimators, max_depth, min_samples_split):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_outcome_train, cv=cv_strategy, scoring='accuracy')
    return scores.mean()

# Random Forest Classifier for Disease
def rf_cv_disease(n_estimators, max_depth, min_samples_split):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_disease_train, cv=cv_strategy, scoring='accuracy')
    return scores.mean()

# K-Nearest Neighbors Classifier for Outcome
def knn_cv_outcome(n_neighbors):
    model = KNeighborsClassifier(
        n_neighbors=int(n_neighbors)
    )
    scores = cross_val_score(model, X_train, y_outcome_train, cv=cv_strategy, scoring='accuracy')
    return scores.mean()

# K-Nearest Neighbors Classifier for Disease
def knn_cv_disease(n_neighbors):
    model = KNeighborsClassifier(
        n_neighbors=int(n_neighbors)
    )
    scores = cross_val_score(model, X_train, y_disease_train, cv=cv_strategy, scoring='accuracy')
    return scores.mean()

# Support Vector Classifier for Outcome
def svc_cv_outcome(C, gamma):
    model = SVC(
        C=C,
        gamma=gamma,
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_outcome_train, cv=cv_strategy, scoring='accuracy')
    return scores.mean()

# Support Vector Classifier for Disease
def svc_cv_disease(C, gamma):
    model = SVC(
        C=C,
        gamma=gamma,
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_disease_train, cv=cv_strategy, scoring='accuracy')
    return scores.mean()

# Decision Tree Classifier for Outcome
def dt_cv_outcome(max_depth, min_samples_split):
    model = DecisionTreeClassifier(
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_outcome_train, cv=cv_strategy, scoring='accuracy')
    return scores.mean()

# Decision Tree Classifier for Disease
def dt_cv_disease(max_depth, min_samples_split):
    model = DecisionTreeClassifier(
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_disease_train, cv=cv_strategy, scoring='accuracy')
    return scores.mean()

# MLP Classifier for Outcome
def mlp_cv_outcome(hidden_layer_sizes, alpha):
    model = MLPClassifier(
        hidden_layer_sizes=(int(hidden_layer_sizes),),
        alpha=alpha,
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_outcome_train, cv=cv_strategy, scoring='accuracy')
    return scores.mean()

# MLP Classifier for Disease
def mlp_cv_disease(hidden_layer_sizes, alpha):
    model = MLPClassifier(
        hidden_layer_sizes=(int(hidden_layer_sizes),),
        alpha=alpha,
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_disease_train, cv=cv_strategy, scoring='accuracy')
    return scores.mean()

# Naive Bayes Classifier for Outcome and Disease (no hyperparameters to optimize)
def nb_cv_outcome():
    model = GaussianNB()
    scores = cross_val_score(model, X_train, y_outcome_train, cv=cv_strategy, scoring='accuracy')
    return scores.mean()

def nb_cv_disease():
    model = GaussianNB()
    scores = cross_val_score(model, X_train, y_disease_train, cv=cv_strategy, scoring='accuracy')
    return scores.mean()

# Step 5: Perform Bayesian Optimization for all Models

# Define parameter bounds for different models
rf_bounds = {'n_estimators': (10, 200), 'max_depth': (1, 20), 'min_samples_split': (2, 10)}
knn_bounds = {'n_neighbors': (1, 50)}
svc_bounds = {'C': (0.1, 10), 'gamma': (0.01, 1)}
dt_bounds = {'max_depth': (1, 20), 'min_samples_split': (2, 10)}
mlp_bounds = {'hidden_layer_sizes': (10, 100), 'alpha': (0.0001, 0.1)}

# Bayesian Optimization for Outcome models
optimizers_outcome = [
    BayesianOptimization(f=rf_cv_outcome, pbounds=rf_bounds, random_state=42, verbose=0),
    BayesianOptimization(f=knn_cv_outcome, pbounds=knn_bounds, random_state=42, verbose=0),
    BayesianOptimization(f=svc_cv_outcome, pbounds=svc_bounds, random_state=42, verbose=0),
    BayesianOptimization(f=dt_cv_outcome, pbounds=dt_bounds, random_state=42, verbose=0),
    BayesianOptimization(f=mlp_cv_outcome, pbounds=mlp_bounds, random_state=42, verbose=0)
]

# Bayesian Optimization for Disease models
optimizers_disease = [
    BayesianOptimization(f=rf_cv_disease, pbounds=rf_bounds, random_state=42, verbose=0),
    BayesianOptimization(f=knn_cv_disease, pbounds=knn_bounds, random_state=42, verbose=0),
    BayesianOptimization(f=svc_cv_disease, pbounds=svc_bounds, random_state=42, verbose=0),
    BayesianOptimization(f=dt_cv_disease, pbounds=dt_bounds, random_state=42, verbose=0),
    BayesianOptimization(f=mlp_cv_disease, pbounds=mlp_bounds, random_state=42, verbose=0)
]

# Optimize all models
for optimizer in optimizers_outcome:
    optimizer.maximize(init_points=5, n_iter=20)

for optimizer in optimizers_disease:
    optimizer.maximize(init_points=5, n_iter=20)

# Step 6: Evaluate Naive Bayes without optimization as it's parameter-free
nb_outcome_score = nb_cv_outcome()
nb_disease_score = nb_cv_disease()

# Step 7: Collect all best models for comparison
best_models_outcome = [opt.max['params'] for opt in optimizers_outcome]
best_models_disease = [opt.max['params'] for opt in optimizers_disease]

# Add Naive Bayes to the list
best_models_outcome.append({"model": "GaussianNB", "score": nb_outcome_score})
best_models_disease.append({"model": "GaussianNB", "score": nb_disease_score})

# Step 8: Initialize best models for training and evaluation
best_models_initialized_outcome = [
    RandomForestClassifier(**{k: int(v) if isinstance(v, float) else v for k, v in params.items()}, random_state=42) if "n_estimators" in params else
    KNeighborsClassifier(**{k: int(v) if isinstance(v, float) else v for k, v in params.items()}) if "n_neighbors" in params else
    SVC(**params, random_state=42) if "C" in params else
    DecisionTreeClassifier(**{k: int(v) if isinstance(v, float) else v for k, v in params.items()}, random_state=42) if "max_depth" in params else
    MLPClassifier(hidden_layer_sizes=(int(params['hidden_layer_sizes']),), alpha=params['alpha'], random_state=42) if "hidden_layer_sizes" in params else
    GaussianNB()
    for params in best_models_outcome
]

best_models_initialized_disease = [
    RandomForestClassifier(**{k: int(v) if isinstance(v, float) else v for k, v in params.items()}, random_state=42) if "n_estimators" in params else
    KNeighborsClassifier(**{k: int(v) if isinstance(v, float) else v for k, v in params.items()}) if "n_neighbors" in params else
    SVC(**params, random_state=42) if "C" in params else
    DecisionTreeClassifier(**{k: int(v) if isinstance(v, float) else v for k, v in params.items()}, random_state=42) if "max_depth" in params else
    MLPClassifier(hidden_layer_sizes=(int(params['hidden_layer_sizes']),), alpha=params['alpha'], random_state=42) if "hidden_layer_sizes" in params else
    GaussianNB()
    for params in best_models_disease
]

# Step 9: Evaluate all best models on training, validation, and test data and save the best one based on test accuracy
best_accuracy_outcome = 0
best_model_outcome = None
best_model_name_outcome = ""
for model in best_models_initialized_outcome:
    model_name = type(model).__name__
    model.fit(X_train, y_outcome_train)
    train_accuracy = accuracy_score(y_outcome_train, model.predict(X_train))
    val_accuracy = accuracy_score(y_outcome_val, model.predict(X_val))
    test_accuracy = accuracy_score(y_outcome_test, model.predict(X_test))
    print(f"{model_name} (Outcome) training accuracy: {train_accuracy:.4f}")
    print(f"{model_name} (Outcome) validation accuracy: {val_accuracy:.4f}")
    print(f"{model_name} (Outcome) test accuracy: {test_accuracy:.4f}")
    if test_accuracy > best_accuracy_outcome:
        best_accuracy_outcome = test_accuracy
        best_model_outcome = model
        best_model_name_outcome = model_name

best_accuracy_disease = 0
best_model_disease = None
best_model_name_disease = ""
for model in best_models_initialized_disease:
    model_name = type(model).__name__
    model.fit(X_train, y_disease_train)
    train_accuracy = accuracy_score(y_disease_train, model.predict(X_train))
    val_accuracy = accuracy_score(y_disease_val, model.predict(X_val))
    test_accuracy = accuracy_score(y_disease_test, model.predict(X_test))
    print(f"{model_name} (Disease) training accuracy: {train_accuracy:.4f}")
    print(f"{model_name} (Disease) validation accuracy: {val_accuracy:.4f}")
    print(f"{model_name} (Disease) test accuracy: {test_accuracy:.4f}")
    if test_accuracy > best_accuracy_disease:
        best_accuracy_disease = test_accuracy
        best_model_disease = model
        best_model_name_disease = model_name

# Step 10: Save the best models with the name of the model
joblib.dump(best_model_outcome, f"{best_model_name_outcome}_outcome_best.pkl")
joblib.dump(best_model_disease, f"{best_model_name_disease}_disease_best.pkl")
print(f"Best model for outcome saved as {best_model_name_outcome}_outcome_best.pkl")
print(f"Best model for disease saved as {best_model_name_disease}_disease_best.pkl")

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 11: Evaluate the best models on the test set and generate confusion matrices
# For Outcome Model
print("\nConfusion Matrix and Classification Report for the Best Outcome Model:")

# Predict on the test set using the best outcome model
y_outcome_test_pred = best_model_outcome.predict(X_test)

# Calculate confusion matrix for the outcome model
conf_matrix_outcome = confusion_matrix(y_outcome_test, y_outcome_test_pred)
print("Confusion Matrix for Outcome:")
print(conf_matrix_outcome)

# Display classification report for the outcome model
print("\nClassification Report for Outcome:")
print(classification_report(y_outcome_test, y_outcome_test_pred))

# Plot confusion matrix for the outcome model using ConfusionMatrixDisplay
plt.figure(figsize=(8, 6))
disp_outcome = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_outcome, display_labels=le.inverse_transform(np.unique(y_outcome_test)))
disp_outcome.plot(cmap='Blues')
plt.title(f"Confusion Matrix for {best_model_name_outcome} (Outcome)")
plt.show()

# For Disease Model
print("\nConfusion Matrix and Classification Report for the Best Disease Model:")

# Predict on the test set using the best disease model
y_disease_test_pred = best_model_disease.predict(X_test)

# Calculate confusion matrix for the disease model
conf_matrix_disease = confusion_matrix(y_disease_test, y_disease_test_pred)
print("Confusion Matrix for Disease:")
print(conf_matrix_disease)

# Display classification report for the disease model
print("\nClassification Report for Disease:")
print(classification_report(y_disease_test, y_disease_test_pred))

# Plot confusion matrix for the disease model using ConfusionMatrixDisplay
plt.figure(figsize=(8, 6))
#disp_disease = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_disease, display_labels=le.inverse_transform(np.unique(y_disease_test)))
#disp_disease.plot(cmap='Blues')
#plt.title(f"Confusion Matrix for {best_model_name_disease} (Disease)")
#plt.show()

# Step 11: Load the Saved Models and Perform Inference with New Data
loaded_model_outcome = joblib.load("/content/SVC_outcome_best.pkl")
loaded_model_disease = joblib.load("/content/RandomForestClassifier_disease_best.pkl")

new_data = pd.DataFrame({
    'Fever': [0],
    'Cough': [1],
    'Fatigue': [0],
    'Difficulty Breathing': [0],
    'Blood Pressure': [1],
    'Cholesterol Level': [0],
    'Gender': [0]
    # Add other one-hot encoded columns if any
    # Add other continuous or categorical variables if present
})

# If any one-hot encoded columns are missing in new data, add them with a value of 0
missing_cols = set(X_train.columns) - set(new_data.columns)
for col in missing_cols:
    new_data[col] = 0

# Ensure the order of columns in new_data matches the training data
new_data = new_data[X_train.columns]

# Perform inference
prediction_outcome = loaded_model_outcome.predict(new_data)
prediction_disease = loaded_model_disease.predict(new_data)
le_disease = joblib.load("le_Disease.pkl")
predicted_disease_name = le_disease.inverse_transform(prediction_disease)

print(f"Prediction for new data (Outcome): {prediction_outcome[0]}")
print(f"Prediction for new data (Disease): {predicted_disease_name[0]}")

"""ChatBot Test"""

!pip install llama-index==0.5.6
!pip install langchain==0.0.148
!pip install openai==0.28

import joblib
import pandas as pd
import re  # Import regular expressions for extracting information
from sklearn.preprocessing import LabelEncoder
import openai
from langchain import OpenAI, LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
import os
import openai

# Load the trained models and label encoder
rf_model_outcome = joblib.load("/content/SVC_outcome_best.pkl")
rf_model_disease = joblib.load("/content/RandomForestClassifier_disease_best.pkl")
le_disease = joblib.load("/content/le_Disease.pkl")  # Assuming LabelEncoder is saved as well

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] =  input("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")  # Replace with your actual API key

# Define a function to extract relevant features from the input text
def extract_features_from_text(user_input):
    # Define the feature keywords and corresponding Yes/No mappings
    feature_keywords = {
        "fever": "Fever",
        "cough": "Cough",
        "fatigue": "Fatigue",
        "difficulty breathing": "Difficulty Breathing",
        "blood pressure": "Blood Pressure",
        "cholesterol level": "Cholesterol Level"
    }

    # Initialize a dictionary to store the extracted features
    extracted_features = {
        "Fever": 0,
        "Cough": 0,
        "Fatigue": 0,
        "Difficulty Breathing": 0,
        "Blood Pressure": "Normal",  # Default value, can be updated to 'High' or 'Low'
        "Cholesterol Level": "Normal"  # Default value, can be updated to 'High' or 'Low'
    }

    # Use regular expressions to extract Yes/No and High/Low features
    for key, value in feature_keywords.items():
        if re.search(rf"\b{key}\b", user_input, re.IGNORECASE):
            if value in ["Blood Pressure", "Cholesterol Level"]:
                # Check for High/Low in the user input
                if re.search(r"\bhigh\b", user_input, re.IGNORECASE):
                    extracted_features[value] = "High"
                elif re.search(r"\blow\b", user_input, re.IGNORECASE):
                    extracted_features[value] = "Low"
                else:
                    extracted_features[value] = "Normal"  # Default to normal if not specified
            else:
                # Check for Yes/No in the user input
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

    # Ensure all required columns are present
    missing_cols = set(X_train.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    # Ensure the order of columns matches the training data
    input_df = input_df[X_train.columns]

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

# Define a tool for LangChain to use for prediction
def prediction_tool(query):
    # Call perform_inference function with the query to get prediction results
    return perform_inference_from_text(query)

# Initialize the prediction tool for the agent
tools = [
    Tool(
        name="Predict Disease Outcome",
        func=prediction_tool,
        description="Use this tool to predict the disease outcome based on user input",
    )
]

# Initialize memory for the conversation
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the OpenAI LLM with the GPT-4 model
llm = OpenAI(temperature=0.7, model="gpt-4")  # Updated model name

# Initialize the LangChain agent with the tools and memory
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)

# Start the chatbot loop
print("Chatbot is ready! Type your symptoms in natural language or 'exit' to stop.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Perform inference using the prediction tool directly
    prediction = perform_inference_from_text(user_input)
    print(f"Chatbot: {prediction}")

!pip install streamlit pyngrok --quiet

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app7.py
# import joblib
# import pandas as pd
# import re
# import streamlit as st
# 
# # Load the trained models and label encoder
# rf_model_outcome = joblib.load("/content/SVC_outcome_best.pkl")
# rf_model_disease = joblib.load("/content/RandomForestClassifier_disease_best.pkl")
# le_disease = joblib.load("/content/le_Disease.pkl")  # Assuming LabelEncoder is saved as well
# # Sample data for X_train columns (replace with actual data structure)
# # Ensure this list matches the exact features used during model training
# X_train_columns = [
#     'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
#     'Age', 'Blood Pressure', 'Cholesterol Level'  # Include all features used during training
# ]
# 
# # Define a function to extract relevant features from the input text
# def extract_features_from_text(user_input):
#     # Define the feature keywords and corresponding Yes/No mappings
#     feature_keywords = {
#         "fever": "Fever",
#         "cough": "Cough",
#         "fatigue": "Fatigue",
#         "difficulty breathing": "Difficulty Breathing",
#         "age": "Age",
#         "blood pressure": "Blood Pressure",
#         "cholesterol level": "Cholesterol Level"# Include additional features here
#     }
# 
#     # Initialize a dictionary to store the extracted features
#     extracted_features = {
#         "Fever": 0,
#         "Cough": 0,
#         "Fatigue": 0,
#         "Difficulty Breathing": 0,
#         "Age": 0,
#         "Blood Pressure": "Normal",  # Default value, can be updated to 'High' or 'Low'
#         "Cholesterol Level": "Normal"  # Default value, can be updated to 'High' or 'Low'
# 
#     }
# 
#     # Use regular expressions to extract Yes/No, High/Low, and numerical features
#     for key, value in feature_keywords.items():
#         if value in ["Blood Pressure", "Cholesterol Level"]:
#             # Check for High/Low in the user input
#             if re.search(r"\bhigh\b", user_input, re.IGNORECASE):
#                 extracted_features[value] = "High"
#             elif re.search(r"\blow\b", user_input, re.IGNORECASE):
#                 extracted_features[value] = "Low"
#             else:
#                 extracted_features[value] = "Normal"  # Default to normal if not specified
#         elif value == "Age":
#             # Extract numerical value for Age
#             age_match = re.search(r'\bage\s+(\d+)\b', user_input, re.IGNORECASE)
#             if age_match:
#                 extracted_features[value] = int(age_match.group(1))
#             else:
#                 extracted_features[value] = 30  # Default value if age not specified
#         else:
#             # Check for Yes/No in the user input for other features
#             if re.search(rf"\b{key}\b.*\byes\b", user_input, re.IGNORECASE):
#                 extracted_features[value] = 1  # Yes
#             elif re.search(rf"\b{key}\b.*\bno\b", user_input, re.IGNORECASE):
#                 extracted_features[value] = 0  # No
# 
#     return extracted_features
# 
# # Define a function to perform inference based on extracted features
# def perform_inference_from_text(user_input):
#     # Extract features from user input text
#     features = extract_features_from_text(user_input)
# 
#     # Convert blood pressure and cholesterol levels to numerical encoding
#     bp_mapping = {"Low": 0, "Normal": 1, "High": 2}
#     chol_mapping = {"Low": 0, "Normal": 1, "High": 2}
# 
#     features['Blood Pressure'] = bp_mapping[features['Blood Pressure']]
#     features['Cholesterol Level'] = chol_mapping[features['Cholesterol Level']]
# 
#     # Convert extracted features to a DataFrame
#     input_df = pd.DataFrame([features])
# 
#     # Ensure all required columns are present by checking against the original feature set
#     missing_cols = set(X_train_columns) - set(input_df.columns)
#     for col in missing_cols:
#         input_df[col] = 0  # Assign default value for missing columns
# 
#     # Ensure the order of columns matches the training data
#     input_df = input_df[X_train_columns]
# 
#     # Perform inference
#     outcome_prediction = rf_model_outcome.predict(input_df)[0]
#     disease_prediction = rf_model_disease.predict(input_df)[0]
#     predicted_disease_name = le_disease.inverse_transform([disease_prediction])[0]
# 
#     # Generate result message
#     result_message = (
#         f"Predicted Outcome: {'Positive' if outcome_prediction == 1 else 'Negative'}\n"
#         f"Predicted Disease: {predicted_disease_name}"
#     )
#     return result_message
# 
# # Initialize the Streamlit app
# st.title("Disease and Outcome Prediction")
# st.write("Enter your symptoms and other health indicators below:")
# 
# # Create input fields in the Streamlit app
# user_input = st.text_area("Describe your symptoms and health indicators in natural language:", "")
# 
# # Perform prediction when the button is clicked
# if st.button("Predict"):
#     if user_input:
#         result = perform_inference_from_text(user_input)
#         st.success(result)
#     else:
#         st.warning("Please enter your symptoms and health indicators.")
#

# Install required packages
!pip install pyngrok streamlit

# Import ngrok and set authtoken
from pyngrok import ngrok

# Set your ngrok authtoken
ngrok.set_auth_token("2kcnlCsJt1HVHYQJSkry71SiHxh_5n6PaTYZLWXu9MyYENHhp")

# Start Streamlit app in the background
!streamlit run app7.py &>/dev/null&

# Create a public URL using ngrok with the correct tunnel configuration
public_url = ngrok.connect(addr="8501")
print(f'Public URL: {public_url}')

# Function to kill all Streamlit processes
import time
import subprocess
from pyngrok import ngrok
import requests
import os
import signal

def kill_all_streamlit_processes():
    result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE)
    process_list = result.stdout.decode('utf-8').split('\n')
    for process in process_list:
        if 'streamlit' in process:
            pid = int(process.split()[1])
            os.kill(pid, signal.SIGKILL)
    print("All Streamlit processes have been terminated.")

# Function to start the Streamlit application
def start_streamlit_app(script_name):
    process = subprocess.Popen(['streamlit', 'run', script_name])
    time.sleep(10)  # Wait for the Streamlit app to start

    # Check if the Streamlit app is running
    try:
        response = requests.get("http://localhost:8501")
        if response.status_code == 200:
            print("Streamlit app is running.")
    except requests.ConnectionError:
        raise RuntimeError("Streamlit app did not start.")

    # Create a public URL using ngrok
    public_url = ngrok.connect(addr="8501")
    print(f'Public URL: {public_url}')
    return process

# Function to close existing ngrok tunnels
def close_existing_tunnels():
    tunnels = ngrok.get_tunnels()
    for tunnel in tunnels:
        ngrok.disconnect(tunnel.public_url)
    print("All ngrok tunnels have been closed.")

# Kill all existing Streamlit processes
kill_all_streamlit_processes()

# Close existing ngrok tunnels
close_existing_tunnels()