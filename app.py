import streamlit as st
import pandas as pd
import joblib

# Set page config - This should be the first Streamlit command
st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="📚",  # Using a book emoji as the icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve aesthetics with pastel colors and contrasting text
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f4f8;  /* Light pastel background */
        color: #1e3a5f;  /* Darker text for better contrast */
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #2c3e50;
    }
    p, label, .stSelectbox, .stRadio {
        font-weight: bold !important;
        color: #34495e;  /* Darker color for labels */
    }
    h1, h2, h3 {
        color: #2980b9;  /* A contrasting blue for headings */
    }
    </style>
    """, unsafe_allow_html=True)

# Load the pre-trained model
model = joblib.load('stress_model.pkl')

# Get the columns used during training directly from the model
trained_columns = model.feature_names_in_

# Function to preprocess input data
def preprocess_input(data, trained_columns):
    categorical_columns = ['Gender', 'Grade Level', 'Part-Time Job', 'Mental Health Issues']
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=False)
    for col in trained_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0
    return data_encoded[trained_columns]

# Streamlit UI
st.title('Student Stress Level Predictor 📊')
st.write("This app predicts the stress level based on various inputs.")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age (18-100)', min_value=18, max_value=100, value=20)
    gender = st.radio('Gender', options=['Male', 'Female'])
    grade_level = st.selectbox('Grade Level', options=['Freshman', 'Sophomore', 'Junior', 'Senior'])
    study_hours = st.number_input('Study Hours per Week (1-40)', min_value=1, max_value=168, value=10)
    sleep_hours = st.number_input('Sleep Hours per Night (4-12)', min_value=4, max_value=12, value=7)
    exercise_freq = st.number_input('Exercise Frequency (times per week, 0-7)', min_value=0, max_value=7, value=3)

with col2:
    social_media = st.number_input('Social Media Usage (hours/day, 0-10)', min_value=0, max_value=10, value=2)
    gpa = st.number_input('Academic Performance GPA (0.0-4.0)', min_value=0.0, max_value=4.0, value=3.0, step=0.1)
    family_support = st.number_input('Family Support (1-5)', min_value=1, max_value=5, value=4)
    social_support = st.number_input('Social Support (1-5)', min_value=1, max_value=5, value=3)
    part_time_job = st.radio('Do you have a part-time job?', options=['Yes', 'No'])
    financial_stress = st.number_input('Financial Stress (1-5)', min_value=1, max_value=5, value=3)
    mental_health_issues = st.radio('Do you have mental health issues?', options=['Yes', 'No'])

# Create a DataFrame with the input values
user_input = pd.DataFrame({
    'Age': [age], 'Gender': [gender], 'Grade Level': [grade_level],
    'Study Hours per Week': [study_hours], 'Sleep Hours per Night': [sleep_hours],
    'Exercise Frequency': [exercise_freq], 'Social Media Use (hrs/day)': [social_media],
    'Academic Performance (GPA)': [gpa], 'Family Support': [family_support],
    'Social Support': [social_support], 'Part-Time Job': [part_time_job],
    'Financial Stress': [financial_stress], 'Mental Health Issues': [mental_health_issues]
})

# Preprocess the user input
user_input_encoded = preprocess_input(user_input, trained_columns)

# Predict button
if st.button('Predict Stress Level'):
    with st.spinner('Predicting...'):
        try:
            # Predict the stress level
            predicted_stress_level = model.predict(user_input_encoded)[0]
            
            # Display the prediction result
            st.subheader(f"Predicted Stress Level: {predicted_stress_level:.2f}")
            
            # Interpret the stress level and provide tips
            if predicted_stress_level < 3:
                interpretation = "Low stress level. You seem to be managing well."
                color = "green"
                tips = """
                Tips to maintain low stress:
                - Continue your current stress management techniques
                - Practice regular self-care activities like exercise and hobbies
                - Use positive self-talk to reinforce good habits
                - Maintain relaxation techniques like deep breathing or meditation
                """
            elif 3 <= predicted_stress_level < 6:
                interpretation = "Moderate stress level. This is common among students. Consider stress management techniques if you feel overwhelmed."
                color = "orange"
                tips = """
                Tips to manage moderate stress:
                - Increase physical activity to release tension
                - Try new relaxation techniques like yoga or tai chi
                - Implement time management strategies to reduce daily pressures
                - Connect more with supportive friends and family
                - Consider cutting back on caffeine and sugar intake
                """
            elif 6 <= predicted_stress_level < 8:
                interpretation = "High stress level. It might be beneficial to seek support or implement stress reduction strategies."
                color = "red"
                tips = """
                Tips to reduce high stress:
                - Prioritize sleep and establish a consistent sleep routine
                - Seek professional help or counseling for additional support
                - Use stress-stopping techniques like counting to 10 or taking breaks
                - Analyze and potentially reduce commitments that cause stress
                - Practice mindfulness or guided imagery to calm your mind
                - Consider lifestyle changes that might be contributing to stress
                """
            else:
                interpretation = "Very high stress level. It's recommended to talk to a counselor or mental health professional for support."
                color = "dark red"
                tips = """
                Tips for very high stress:
                - Seek immediate professional help or counseling
                - Prioritize self-care and stress reduction activities
                - Communicate with trusted friends, family, or mentors about your stress
                - Consider temporary adjustments to your workload or commitments
                - Practice relaxation techniques regularly (e.g., deep breathing, meditation)
                - Ensure you're getting adequate sleep and nutrition
                """
            
            st.markdown(f"<h3 style='color: {color};'>{interpretation}</h3>", unsafe_allow_html=True)
            st.info(tips)
            
            # Disclaimer section
            st.warning("""
                **Disclaimer:** 
                The predicted stress level score is based on a machine learning model and is not 100% accurate. 
                This tool is intended for informational purposes only and should not be used as a substitute for professional advice.
                Please interpret the score as a general indication of potential stress levels rather than an absolute measure.
                If you are experiencing significant stress or mental health concerns, consider reaching out to a qualified professional.
            """)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Please check if all input fields are filled correctly.")
