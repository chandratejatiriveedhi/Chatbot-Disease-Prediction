# Chatbot-Disease-Prediction
The code files in python directory consists of the Jupyter Notebook used to train ML models for predictions of the disease and its outcomes based on the user input. Jupyter notebook files consist of exploratory data analysis, ML model training, testing and validation processes. Additionally, there is a python script version of this notebook. You need your api key to run a chatbot in this notebook. Additionally, csv file of the dataset used is uploaded in this folder. 


The code files in app consist of streamlit app script. To run the script place all the files in same directory and run the following commands. Additionally, pickle files of the trained ML models to generate outcomes of the user prompt input, is provided here. 

!pip install pyngrok streamlit
!streamlit run app7.py &>/dev/null&

Once the application opens you can input in the textbox in the following format. 
Fever: Yes, Cough: No, Fatigue: Yes, Difficulty Breathing: Yes, Age: 45, Blood Pressure: High, Cholesterol Level: Low
