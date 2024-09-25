import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset (e.g., Framingham Heart Study dataset)
data = pd.read_csv('framingham_heart_study.csv')

# Preprocess data (handling missing values, encoding categorical data)
data.dropna(inplace=True)

# Define features and target variable
X = data[['age', 'gender', 'sysBP', 'diaBP', 'BMI', 'cholesterol', 'diabetes', 'smoker']]
y = data['heart_disease']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Test accuracy of the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model using pickle
import pickle
with open('health_risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)
3. Create the Streamlit Web App:
Streamlit will allow us to build a simple, interactive web interface where users can input their health data and receive risk predictions.

Here’s a basic structure for the Streamlit app:

python
Copy code
import streamlit as st
import pandas as pd
import numpy as np
import pickle
# Load the trained model
with open('health_risk_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the web app
st.title("Health Risk Calculator")

# Collect user input
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
sysBP = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
diaBP = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=130, value=80)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=180)
diabetes = st.selectbox("Do you have diabetes?", ["No", "Yes"])
smoker = st.selectbox("Are you a smoker?", ["No", "Yes"])

# Convert user input into a format the model can understand
input_data = pd.DataFrame([[age, 1 if gender == "Male" else 0, sysBP, diaBP, bmi, cholesterol, 1 if diabetes == "Yes" else 0, 1 if smoker == "Yes" else 0]],
                          columns=['age', 'gender', 'sysBP', 'diaBP', 'BMI', 'cholesterol', 'diabetes', 'smoker'])

# Make predictions using the loaded model
risk_prediction = model.predict(input_data)[0]
risk_probability = model.predict_proba(input_data)[0][1]

# Display the result
if st.button("Calculate Health Risk"):
    st.write(f"Your predicted risk of heart disease is: {'High' if risk_prediction else 'Low'}")
    st.write(f"Risk probability: {risk_probability:.2f}"
             Data Storage: Use SQL to store user input and prediction data for future analysis.

Example:

python
Copy code
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('health_risk_data.db')
c = conn.cursor()

# Create table for storing user input and predictions
c.execute('''
    CREATE TABLE IF NOT EXISTS risk_predictions (
        id INTEGER PRIMARY KEY,
        age INTEGER,
        gender TEXT,
        sysBP INTEGER,
        diaBP INTEGER,
        bmi REAL,
        cholesterol INTEGER,
        diabetes INTEGER,
        smoker INTEGER,
        risk_prediction INTEGER,
        risk_probability REAL
    )
''')

# Insert user data and prediction
c.execute('''
    INSERT INTO risk_predictions (age, gender, sysBP, diaBP, bmi, cholesterol, diabetes, smoker, risk_prediction, risk_probability)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', (age, gender, sysBP, diaBP, bmi, cholesterol, diabetes, smoker, risk_prediction, risk_probability))

# Commit and close connection
conn.commit()
conn.close()
5. Deployment to Azure:
Once the app is ready, you can deploy it to Azure App Service.

Steps to deploy:

Create a Dockerfile to containerize the Streamlit app:

Dockerfile
Copy code
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
Deploy the container to Azure App Service.

6. Requirements.txt:
Make sure you have a requirements.txt file to specify dependencies:

makefile
Copy code
streamlit==1.14.0
scikit-learn==1.0.2
pandas==1.3.3
numpy==1.21.2
sqlite3
app.py (Streamlit app):
python
Copy code
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('health_risk_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Health Risk Calculator")

# User inputs
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
sysBP = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
diaBP = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=130, value=80)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=180)
diabetes = st.selectbox("Do you have diabetes?", ["No", "Yes"])
smoker = st.selectbox("Are you a smoker?", ["No", "Yes"])

# Prepare input data for prediction
input_data = pd.DataFrame([[age, 1 if gender == "Male" else 0, sysBP, diaBP, bmi, cholesterol, 1 if diabetes == "Yes" else 0, 1 if smoker == "Yes" else 0]],
                          columns=['age', 'gender', 'sysBP', 'diaBP', 'BMI', 'cholesterol', 'diabetes', 'smoker'])

# Make predictions
risk_prediction = model.predict(input_data)[0]
risk_probability = model.predict_proba(input_data)[0][1]

if st.button("Calculate Health Risk"):
    st.write(f"Your predicted risk of heart disease is: {'High' if risk_prediction else 'Low'}")
    st.write(f"Risk probability: {risk_probability:.2f}")
model_training.py (Model Training):
python
Copy code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv('framingham_heart_study.csv')

# Preprocess data
data.dropna(inplace=True)
# Define features and target
X = data[['age', 'gender', 'sysBP', 'diaBP', 'BMI', 'cholesterol', 'diabetes', 'smoker']]
y = data['heart_disease']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('health_risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)
requirements.txt:
makefile
Copy code
streamlit==1.14.0
pandas==1.3.3
scikit-learn==1.0.2
numpy==1.21.2
pickle5
Dockerfile (optional, for deployment):
Dockerfile


