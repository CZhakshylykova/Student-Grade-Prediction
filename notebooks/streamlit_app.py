import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Set page configuration and title
st.set_page_config(page_title='Student Performance Prediction', layout='centered')

# Cache the model loading
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'regression_model.pkl')   
    model = joblib.load(model_path)
    return model

model = load_model()

# App title
st.title(":orange[Student Performance Prediction]")  # Title spans across the whole page


# Split the page into two columns
col1, col2 = st.columns([1, 2])  # Split into two columns: col1 takes 1/3, col2 takes 2/3

# Left column: About the study
with col1:
    st.header("About the Study")
    st.write("""
    This study explores various factors that affect student performance, 
    with the goal of predicting their final grades based on input features. 
    Some of the key features include:
    - Mother's education level
    - Alcohol consumption on workdays and weekends
    - Number of school absences
    - First and second period grades
    - Participation in extracurricular activities

    Link to Dataset: https://archive.ics.uci.edu/dataset/320/student+performance
    
    This project is a collaborative initiative brought by SuperDataScience community

 """)

# Right column: Feature input
with col2:

    st.header("Input Features")  # Feature input title is now part of the input container

    Medu = st.selectbox("Mother's Education (0: None, 1: Primary, 2: 5th-9th, 3: Secondary, 4: Higher)", [0, 1, 2, 3, 4])
    failures = st.select_slider("Past Class Failures", options=[0, 1, 2, 3])
    Dalc = st.select_slider("Alcohol Consumption on Workdays (1: Very Low, 5: Very High)", options=[1, 2, 3, 4, 5])
    Walc = st.select_slider("Alcohol Consumption on Weekends (1: Very Low, 5: Very High)", options=[1, 2, 3, 4, 5])
    absences = st.number_input("Number of School Absences (0-93)", min_value=0, max_value=93, step=1)
    G1 = st.number_input("First Period Grade (0-20)", min_value=0, max_value=20, step=1)
    G2 = st.number_input("Second Period Grade (0-20)", min_value=0, max_value=20, step=1)
    sex = st.selectbox("Gender (F: Female, M: Male)", ['F', 'M'])
    Mjob = st.selectbox("Mother's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
    schoolsup = st.selectbox("Extra Educational Support (yes/no)", ['yes', 'no'])
    famsup = st.selectbox("Family Educational Support (yes/no)", ['yes', 'no'])
    activities = st.selectbox("Extra-curricular Activities (yes/no)", ['yes', 'no'])
    nursery = st.selectbox("Attended Nursery School (yes/no)", ['yes', 'no'])
    higher = st.selectbox("Aiming for Higher Education (yes/no)", ['yes', 'no'])

    # Columns definition for model prediction
    columns = ['Medu', 'failures', 'Dalc', 'Walc', 'absences', 'G1', 'G2', 'sex', 'Mjob', 'schoolsup', 'famsup', 'activities', 'nursery', 'higher']

    def preprocess_input(Medu, failures, Dalc, Walc, absences, G1, G2, sex, Mjob, schoolsup, famsup, activities, nursery, higher):
        # Encode categorical variables
        sex_encoded = 1 if sex == 'M' else 0
        schoolsup_encoded = 1 if schoolsup == 'yes' else 0
        famsup_encoded = 1 if famsup == 'yes' else 0
        nursery_encoded = 1 if nursery == 'yes' else 0
        higher_encoded = 1 if higher == 'yes' else 0
        activities_encoded = 1 if activities == 'yes' else 0
        Mjob_encoded = ['teacher', 'health', 'services', 'at_home', 'other'].index(Mjob)
        
        # Construct input array
        row = np.array([Medu, failures, Dalc, Walc, absences, G1, G2, sex_encoded, Mjob_encoded, 
                        schoolsup_encoded, famsup_encoded, activities_encoded, nursery_encoded, higher_encoded])
        return pd.DataFrame([row], columns=columns)

    # Prediction function
    def predict():
        X = preprocess_input(Medu, failures, Dalc, Walc, absences, G1, G2, sex, Mjob, schoolsup, famsup, activities, nursery, higher)
        with st.spinner("Making prediction..."):
            prediction = model.predict(X)[0]
        st.success(f"Predicted Final Grade: {prediction:.2f}")

    # Prediction button
    if st.button("Predict"):
        predict()