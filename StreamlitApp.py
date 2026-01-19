import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==========================================
# 1. Load the Model
# ==========================================
@st.cache_resource
def load_model():
    with open('best_health_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'best_health_model.pkl' not found. Please run the training script first.")
    st.stop()

# ==========================================
# 2. Streamlit UI Layout
# ==========================================
st.title("🏥 Health Score Predictor")
st.write("Enter the patient's details below to predict their Overall Health Score.")

# Create a form for user input
with st.form("health_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        calories = st.number_input("Average Daily Calories", min_value=500, max_value=10000, value=2000)
        fast_food = st.number_input("Fast Food Meals per Week", min_value=0, max_value=50, value=2)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5)

    with col2:
        activity = st.number_input("Physical Activity (Hours/Week)", min_value=0.0, max_value=168.0, value=3.0)
        sleep = st.number_input("Sleep (Hours/Day)", min_value=0.0, max_value=24.0, value=7.5)
        energy = st.slider("Energy Level Score (1-10)", 1, 10, 5)
        digestive = st.selectbox("Digestive Issues", ["No", "Yes"])
        doctor_visits = st.number_input("Doctor Visits per Year", min_value=0, max_value=100, value=1)

    # Submit button
    submitted = st.form_submit_button("Predict Health Score")

# ==========================================
# 3. Prediction Logic
# ==========================================
if submitted:
    # Organize features into a dictionary (matching the training data column names)
    input_data = {
        'Age': age,
        'Gender': gender,
        'Fast_Food_Meals_Per_Week': fast_food,
        'Average_Daily_Calories': calories,
        'BMI': bmi,
        'Physical_Activity_Hours_Per_Week': activity,
        'Sleep_Hours_Per_Day': sleep,
        'Energy_Level_Score': energy,
        'Digestive_Issues': digestive,
        'Doctor_Visits_Per_Year': doctor_visits
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        # Display the result with some color formatting based on score
        if prediction >= 7:
            st.success(f"Predicted Overall Health Score: **{prediction:.2f} / 10**")
        elif prediction >= 4:
            st.warning(f"Predicted Overall Health Score: **{prediction:.2f} / 10**")
        else:
            st.error(f"Predicted Overall Health Score: **{prediction:.2f} / 10**")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")