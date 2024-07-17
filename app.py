import streamlit as st
import pickle
import numpy as np

# Load the model
with open('rf_classifier.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Define a function to preprocess and predict
def predict_stroke(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status):
    # Convert categorical variables to numerical
    gender_encoded = 1 if gender == 'Male' else 0
    ever_married_encoded = 1 if ever_married == 'Yes' else 0

    # Encode work_type
    if work_type == 'Private':
        work_type_encoded = 0
    elif work_type == 'Self-employed':
        work_type_encoded = 1
    elif work_type == 'Govt_job':
        work_type_encoded = 2
    elif work_type == 'Children':
        work_type_encoded = 3
    else:  # 'Never_worked'
        work_type_encoded = 4

    # Encode residence_type
    residence_type_encoded = 1 if residence_type == 'Urban' else 0

    # Encode smoking_status
    if smoking_status == 'smokes':
        smoking_status_encoded = 0
    elif smoking_status == 'formerly_smoked':
        smoking_status_encoded = 1
    else:  # 'never_smoked'
        smoking_status_encoded = 2

    # Create feature array
    features = np.array([gender_encoded, age, hypertension, heart_disease, ever_married_encoded,
                         work_type_encoded, residence_type_encoded, avg_glucose_level, bmi, smoking_status_encoded]).reshape(1, -1)

    # Predict using the model
    prediction = rf_model.predict(features)[0]

    return prediction

# Streamlit UI
def main():
    st.title('Stroke Prediction App')

    # Sidebar inputs
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    age = st.slider('Age', 0, 100, 50)
    hypertension = st.select_slider('Hypertension (0 for No, 1 for Yes)', options=[0, 1])
    heart_disease = st.select_slider('Heart Disease (0 for No, 1 for Yes)', options=[0, 1])
    ever_married = st.radio('Ever Married?', ['Yes', 'No'])
    work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'])
    residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
    avg_glucose_level = st.slider('Average Glucose Level', 0.0, 300.0, 100.0)
    bmi = st.slider('BMI', 0.0, 60.0, 25.0)
    smoking_status = st.selectbox('Smoking Status', ['smokes', 'formerly_smoked', 'never_smoked'])

    # Prediction button
    if st.button('Predict'):
        prediction = predict_stroke(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status)
        if prediction == 0:
            st.write('Prediction:', 'High Chance of Stroke')
        else:
            st.write('Prediction:', 'Low Chance of Stroke')

if __name__ == '__main__':
    main()
