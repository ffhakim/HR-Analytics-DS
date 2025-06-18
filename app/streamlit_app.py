import streamlit as st
import pandas as pd
import joblib
import numpy as np
import hashlib
import glob
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Job Change Prediction & Retention AI (v2)",
    page_icon="🚀",
    layout="wide"
)

# authentication 
def hash_password(password):
    """Hashes a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    """Authenticates a user against a predefined dictionary of users."""
    users = {
        "admin": "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f", # password is "password123"
        "user": hash_password("user123") 
    }
    hashed_input = hash_password(password)
    return username in users and users[username] == hashed_input

def login_page():
    """Displays the login form and handles authentication."""
    st.title("🔐 Login to HR Analytics Dashboard")
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun() 
            else:
                st.error("Incorrect username or password.")
    st.info("Demo credentials: username `admin`, password `password123`")

# Artifact joblib Loading 
@st.cache_resource
def load_prediction_artifacts():
    """
    Loads the trained prediction model, scaler, label encoders, and feature columns.
    Looks for files like '*_model.joblib', 'scaler.joblib', etc.
    """
    artifacts = {'model': None, 'model_name': None, 'scaler': None, 'encoders': None, 'columns': None}
    errors = []

    model_files = glob.glob("*_model.joblib")
    if not model_files:
        errors.append("No model file found (e.g., 'RandomForest_model.joblib'). Ensure it's in the root directory.")
    else:
        # Use the first model file found
        model_file = model_files[0]
        try:
            artifacts['model'] = joblib.load(model_file)
            artifacts['model_name'] = os.path.basename(model_file).replace('_model.joblib', '').replace('_', ' ')
        except Exception as e:
            errors.append(f"Error loading prediction model '{model_file}': {e}")

    # Load scaler
    scaler_file = 'scaler.joblib'
    if not os.path.exists(scaler_file):
        errors.append(f"Scaler file ('{scaler_file}') not found. This is required for preprocessing.")
    else:
        try:
            artifacts['scaler'] = joblib.load(scaler_file)
        except Exception as e:
            errors.append(f"Error loading scaler '{scaler_file}': {e}")

    # Load label encoders
    encoders_file = 'label_encoders.joblib'
    if not os.path.exists(encoders_file):
        errors.append(f"Label encoders file ('{encoders_file}') not found. This is required for categorical features.")
    else:
        try:
            artifacts['encoders'] = joblib.load(encoders_file)
        except Exception as e:
            errors.append(f"Error loading label encoders '{encoders_file}': {e}")

    # Load feature columns
    columns_file = 'feature_columns.joblib'
    if not os.path.exists(columns_file):
        errors.append(f"Feature columns file ('{columns_file}') not found. This defines the expected input features.")
    else:
        try:
            artifacts['columns'] = joblib.load(columns_file)
        except Exception as e:
            errors.append(f"Error loading feature columns '{columns_file}': {e}")
            
    # checking model, scaler, encoders, and columns 
    if not all([artifacts['model'], artifacts['scaler'], artifacts['encoders'], artifacts['columns']]):
        st.session_state.artifacts_loaded_successfully = False
    else:
        st.session_state.artifacts_loaded_successfully = True
        
    return artifacts, errors

# Load Hugging Face Model 
@st.cache_resource
def load_llm_model():
    """
    Loads the Hugging Face model for text generation.
    Using FLAN-T5 base model which is good for instruction following and generation.
    """
    try:
        model_name = "google/flan-t5-base"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create a text generation pipeline
        generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=400,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            device=0 if torch.cuda.is_available() else -1  
        )
        
        return generator, None
    except Exception as e:
        error_msg = f"Error loading Hugging Face model: {e}"
        return None, error_msg

# LLM Recommendation (Hugging Face) 
def llm_retention(generator, employee_profile_summary, prediction_details):
    """
    Generates retention strategies using a loaded Hugging Face pipeline.

    Args:
        generator (pipeline): The loaded Hugging Face text generation pipeline.
        employee_profile_summary (dict): A dictionary containing key details of the employee.
        prediction_details (str): A string summarizing the prediction outcome.

    Returns:
        str: AI-generated retention strategies, or an error message.
    """
    if not generator:
        return "AI Advisor Error: Model not loaded."

    prompt = f"""An employee is {prediction_details}. 

Employee Profile:
- City Development Index: {employee_profile_summary.get('city_development_index', 'N/A')}
- Gender: {employee_profile_summary.get('gender', 'N/A')}
- Experience: {employee_profile_summary.get('relevant_experience', 'N/A')}
- Education: {employee_profile_summary.get('education_level', 'N/A')}
- Total Experience: {employee_profile_summary.get('experience', 'N/A')} years
- Company Size: {employee_profile_summary.get('company_size', 'N/A')}
- Training Hours: {employee_profile_summary.get('training_hours', 'N/A')}

Provide 3-5 specific retention strategies for management. Focus on career development, recognition, compensation, and work-life balance. Start each with an action verb."""

    try:
        # generate response
        with st.spinner("generating retention strategies..."):
            response = generator(prompt, max_length=300, num_return_sequences=1)
            
            if response and len(response) > 0:
                generated_text = response[0]['generated_text']
                
                strategies = generated_text.strip()
                
                if not any(char.isdigit() and char in strategies[:10] for char in strategies):
                    sentences = [s.strip() for s in strategies.split('.') if s.strip() and len(s.strip()) > 10]
                    if sentences:
                        formatted_strategies = []
                        for i, sentence in enumerate(sentences[:5], 1): # generate max 5 retention
                            if not sentence.endswith('.'):
                                sentence += '.'
                            formatted_strategies.append(f"{i}. {sentence}")
                        return '\n'.join(formatted_strategies)
                
                return strategies
            else:
                return "AI Advisor could not generate specific recommendations. Please try again."
            
    except Exception as e:
        return f"AI Advisor encountered an error during generation: {str(e)}"

# fallback function if LLM failed
def simple_rule_based_retention(employee_profile_summary):
    """
    A simple rule-based system as fallback when LLM fails.
    """
    strategies = []
    
    # experience based recommendations
    exp_years = employee_profile_summary.get('experience', 'N/A')
    if exp_years in ['<1', '1', '2']:
        strategies.append("Implement a comprehensive mentorship program for junior employees.")
    elif exp_years in ['>20', '20', '19', '18']:
        strategies.append("Offer senior-level leadership opportunities and knowledge sharing roles.")
    
    # education based recommendations
    education = employee_profile_summary.get('education_level', '')
    if 'Masters' in education or 'Phd' in education:
        strategies.append("Provide advanced research projects and thought leadership opportunities.")
    elif 'High School' in education:
        strategies.append("Offer tuition reimbursement and continuing education programs.")
    
    # training based recommendations
    training_hours = employee_profile_summary.get('training_hours', 0)
    try:
        training_hours = float(training_hours)
        if training_hours < 20:
            strategies.append("Increase professional development budget and training opportunities.")
        elif training_hours > 100:
            strategies.append("Recognize the employee's commitment to learning with advanced certifications.")
    except (ValueError, TypeError):
        pass
    
    company_size = employee_profile_summary.get('company_size', '')
    if company_size in ['<10', '10/49']:
        strategies.append("Provide equity opportunities and flexible work arrangements common in startups.")
    elif company_size in ['10000+', '5000-9999']:
        strategies.append("Create clear career advancement paths and cross-functional project opportunities common in large corporations.")
    
    if len(strategies) < 3:
        default_strategies = [
            "Conduct regular one-on-one career development discussions.",
            "Review and adjust compensation package to market standards.",
            "Implement flexible work arrangements and improve work-life balance.",
            "Establish clear performance metrics and recognition programs.",
            "Provide challenging projects aligned with career goals."
        ]
        for strategy in default_strategies:
            if strategy not in strategies and len(strategies) < 5:
                strategies.append(strategy)
    
    return '\n'.join([f"{i+1}. {strategy}" for i, strategy in enumerate(strategies[:5])])

# Data Preprocessing 
def preprocess_input(input_df, encoders, scaler, expected_columns):
    """
    Preprocesses the input DataFrame using loaded artifacts (encoders, scaler).
    Ensures the DataFrame matches the structure expected by the model.
    """
    processed_df = input_df.copy()

    for col, le in encoders.items():
        if col in processed_df.columns:
            known_labels = list(le.classes_)
            processed_df[col] = processed_df[col].apply(
                lambda x: le.transform([x])[0] if x in known_labels else -1 
            )
            if -1 in processed_df[col].unique():
                st.warning(f"Column '{col}' contained unseen values. They were encoded as -1. Prediction accuracy might be affected.")
        else:
            raise ValueError(f"Input data is missing an expected column for encoding: {col}")
            
    # process numeric columns
    for col in expected_columns:
        if col not in encoders: 
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                if processed_df[col].isnull().any():
                    fill_value = 0 
                    processed_df[col].fillna(fill_value, inplace=True)
                    st.warning(f"Numeric column '{col}' had missing values, which were filled with {fill_value}.")
            else:
                    raise ValueError(f"Input data is missing an expected numeric column: {col}")

    for col in expected_columns:
        if col not in processed_df.columns:
            st.warning(f"Feature '{col}' was missing from input, defaulting to 0. This might impact prediction accuracy.")
            processed_df[col] = 0

    try:
        processed_df = processed_df[expected_columns]
    except KeyError as e:
        missing_cols = set(expected_columns) - set(processed_df.columns)
        extra_cols = set(processed_df.columns) - set(expected_columns)
        error_msg = (
            f"Column mismatch before scaling. Error: {e}. "
            f"Expected: {expected_columns}. Got: {list(processed_df.columns)}. "
            f"Missing from input: {missing_cols}. Extra in input: {extra_cols}."
        )
        raise ValueError(error_msg)

    # scaling features
    scaled_data = scaler.transform(processed_df)
    return scaled_data

# main app
def main_app_page():
    """Renders the main application page for prediction and recommendations."""
    st.title("💼 Job Change Prediction & AI Retention Advisor (v2)")

    with st.sidebar:
        st.write(f"Hello, **{st.session_state.get('username', 'Guest')}**!")
        if st.button("Logout", key="logout_button"):
            st.session_state.logged_in = False
            st.session_state.username = None
            for key in list(st.session_state.keys()):
                if key not in ['logged_in', 'username']:
                    del st.session_state[key]
            st.success("Logged out successfully.")
            st.rerun()
        st.markdown("---")

    # Load joblib artifacts
    artifacts, load_errors = load_prediction_artifacts()

    if not st.session_state.get('artifacts_loaded_successfully', False) or load_errors:
        st.error("❌ Core prediction components failed to load. The application cannot proceed.")
        for error in load_errors:
            st.warning(f"- {error}")
        st.info("Ensure `train_model.py` has been run correctly and all required `.joblib` files are present.")
        return

    # Unpack prediction artifacts
    model = artifacts['model']
    model_name = artifacts['model_name']
    scaler = artifacts['scaler']
    encoders = artifacts['encoders']
    feature_columns = artifacts['columns']

    st.success(f"✅ Prediction Model ({model_name}) & Preprocessing Tools loaded successfully!")
    
    # Load LLM model and handle potential errors
    generator, llm_error = load_llm_model()
    if llm_error:
        st.warning(f"LLM Based Engine failed to load: {llm_error}")
        st.info("Rule-based recommendations")
    else:
        st.success("LLM Based Engine loaded successfully!")

    
    with st.expander("📊 Model Information", expanded=False):
        st.write(f"**Prediction Model Type:** {model_name}")
        st.write(f"**AI Recommendation Model:** {'google/flan-t5-base' if not llm_error else 'N/A - Using Rule-Based Fallback'}")
        st.write(f"**Features Expected by Model (Order Matters):** {', '.join(feature_columns)}")
        
    
    st.header("Predict Individual Employee Job Change")
    st.subheader("Enter Employee Information:")

    def get_dynamic_options(encoder_key, default_options):
        if encoders and encoder_key in encoders:
            return list(encoders[encoder_key].classes_)
        return default_options

    exp_col_name = 'relevent_experience' if 'relevent_experience' in feature_columns else 'relevant_experience'
    
    with st.form("prediction_form_single"):
        st.markdown("#### Employee Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.selectbox("City", options=get_dynamic_options('city', ['city_103']), help="Employee's city.")
            city_dev_index = st.slider("City Development Index", 0.0, 1.0, 0.75, 0.001, format="%.3f", help="Index of the city's development.")
            gender = st.selectbox("Gender", options=get_dynamic_options('gender', ['Male', 'Female']), help="Employee's gender.")
        with col2:
            relevant_experience_input = st.selectbox(f"Relevant Experience", options=get_dynamic_options(exp_col_name, ['Has relevent experience']), help="Does the employee have relevant experience?")
            education_level = st.selectbox("Education Level", options=get_dynamic_options('education_level', ["Graduate"]), help="Highest education level.")
            major_discipline = st.selectbox("Major Discipline", options=get_dynamic_options('major_discipline', ["STEM"]), help="Major field of study.")
        with col3:
            experience = st.selectbox("Total Professional Experience (Years)", options=get_dynamic_options('experience', ["10"]), help="Total years of experience.")
            enrolled_university = st.selectbox("Current University Enrollment", options=get_dynamic_options('enrolled_university', ['no_enrollment']), help="Enrollment status in university.")
            company_size = st.selectbox("Current Company Size", options=get_dynamic_options('company_size', ["50-99"]), help="Size of the employer.")
            company_type = st.selectbox("Current Company Type", options=get_dynamic_options('company_type', ["Pvt Ltd"]), help="Type of employer.")
            last_new_job = st.selectbox("Years Since Last Job Change", options=get_dynamic_options('last_new_job', ["1"]), help="Time since last job change.")
            training_hours = st.number_input("Training Hours Completed (Last Year)", min_value=0, max_value=1000, value=50, step=1, help="Total training hours last year.")

        submit_pred_button = st.form_submit_button("Predict & Get AI Recommendations", type="primary", use_container_width=True)

        if submit_pred_button:
            input_data_dict = {
                'city': city, 'city_development_index': city_dev_index, 'gender': gender,
                exp_col_name: relevant_experience_input, 'enrolled_university': enrolled_university,
                'education_level': education_level, 'major_discipline': major_discipline, 'experience': experience,
                'company_size': company_size, 'company_type': company_type, 'last_new_job': last_new_job,
                'training_hours': training_hours
            }
            
            current_employee_profile_for_llm = input_data_dict.copy()
            filtered_input_data = {key: [value] for key, value in input_data_dict.items() if key in feature_columns}
            
            if len(filtered_input_data.keys()) != len(feature_columns):
                missing_form_features = set(feature_columns) - set(filtered_input_data.keys())
                st.error(f"Critical Error: The form is missing inputs for the following model features: {', '.join(missing_form_features)}.")
                return

            input_df = pd.DataFrame(filtered_input_data)

            st.markdown("---")
            st.subheader("📈 Prediction Results & AI Recommendations")

            try:
                processed_data = preprocess_input(input_df.copy(), encoders, scaler, feature_columns)
                prediction_val = model.predict(processed_data)[0]
                prediction_proba_val = model.predict_proba(processed_data) if hasattr(model, 'predict_proba') else None

                res_col1, res_col2 = st.columns(2)
                is_at_risk = (prediction_val == 1)
                
                with res_col1:
                    if is_at_risk:
                        st.error("🚨 **HIGH RISK: LIKELY TO CHANGE JOB**")
                        prediction_summary_for_llm = "high risk of job change"
                    else:
                        st.success("✅ **LOW RISK: LIKELY TO STAY**")
                        prediction_summary_for_llm = "low risk of job change"
                
                with res_col2:
                    if prediction_proba_val is not None:
                        prob_stay, prob_leave = prediction_proba_val[0][0], prediction_proba_val[0][1]
                        st.metric("Probability of Staying", f"{prob_stay:.1%}")
                        st.metric("Probability of Leaving", f"{prob_leave:.1%}")
                        st.progress(prob_leave if is_at_risk else prob_stay)
                        prediction_summary_for_llm += f" (probability of leaving: {prob_leave:.0%})"
                    else:
                        st.info("Probability scores not available for this model.")

                if is_at_risk:
                    st.markdown("---")
                    st.subheader("Suggested Retention Strategies")
                    
                    if not llm_error:
                        st.markdown("**Suggested Actions from AI Advisor (FLAN-T5):**")
                        ai_recommendations = llm_retention(generator, current_employee_profile_for_llm, prediction_summary_for_llm)
                        st.markdown(ai_recommendations.replace(". ", ".\n- ").replace("- ", "\n- "))
                    else:
                        st.markdown("**Fallback - Rule-based Recommendations:**")
                        fallback_recommendations = simple_rule_based_retention(current_employee_profile_for_llm)
                        st.markdown(fallback_recommendations)

            except Exception as e:
                st.error(f"❌ An unexpected error occurred during prediction: {e}")
                st.exception(e)

    # Batch Prediction (CSV Upload) 
    st.markdown("---")
    st.header("📁 Batch Predictions (Upload CSV)")
    csv_columns_list_str = ", ".join(feature_columns)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_uploader")

    if uploaded_file is not None:
        try:
            batch_df_original = pd.read_csv(uploaded_file)
            st.write("**Uploaded Data Preview:**")
            st.dataframe(batch_df_original.head())

            missing_cols_batch = set(feature_columns) - set(batch_df_original.columns)
            if missing_cols_batch:
                st.error(f"Uploaded CSV is missing required columns: {', '.join(missing_cols_batch)}")
            else:
                if st.button("🚀 Process Batch Predictions", key="batch_process_button", use_container_width=True):
                    with st.spinner("Processing batch..."):
                        try:
                            batch_df_to_process = batch_df_original[feature_columns].copy()
                            batch_processed_data = preprocess_input(batch_df_to_process, encoders, scaler, feature_columns)
                            batch_predictions = model.predict(batch_processed_data)
                            
                            results_df = batch_df_original.copy()
                            results_df['Predicted_Job_Change (1=Leave, 0=Stay)'] = batch_predictions
                            
                            if hasattr(model, 'predict_proba'):
                                batch_pred_proba = model.predict_proba(batch_processed_data)
                                results_df['Probability_Leave'] = [p[1] for p in batch_pred_proba]
                                
                                # --- PERUBAHAN DIMULAI DI SINI ---
                                # Mengurutkan DataFrame berdasarkan probabilitas untuk leave (tertinggi ke terendah)
                                results_df = results_df.sort_values(by='Probability_Leave', ascending=False)
                                # --- PERUBAHAN SELESAI ---

                            # --- PERUBAHAN PADA JUDUL ---
                            st.write("**Batch Prediction Results (Sorted by Highest Probability to Leave):**")
                            st.dataframe(results_df)
                            
                            csv_export = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(label="📥 Download Predictions as CSV", data=csv_export, file_name='batch_predictions_sorted.csv', mime='text/csv')
                        except Exception as e:
                            st.error(f"Error during batch prediction: {e}")
        except Exception as e:
            st.error(f"Error reading or processing CSV: {e}")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        main_app_page()
    else:
        login_page()

if __name__ == "__main__":
    main()