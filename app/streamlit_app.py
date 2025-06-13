import streamlit as st
import pandas as pd
import joblib
import hashlib
import glob
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Authentication functions ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    users = {
        "admin": "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f", # secret123
        "user": "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f"
    }
    hashed_input = hash_password(password)
    return username in users and users[username] == hashed_input

def login():
    st.title("🔐 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login berhasil!")
                st.rerun()
            else:
                st.error("Username atau password salah.")
    st.info("Demo credentials: admin/secret123 or user/secret123")

# --- Load model and artifacts ---
@st.cache_resource
def load_artifacts():
    model_files = glob.glob("*_model.joblib")
    scaler_file = 'scaler.joblib'
    encoders_file = 'label_encoders.joblib'
    columns_file = 'feature_columns.joblib'

    artifacts = {'model': None, 'model_name': None, 'scaler': None, 'encoders': None, 'columns': None}
    errors = []

    if not model_files:
        errors.append("Tidak ditemukan file model.")
    else:
        try:
            artifacts['model'] = joblib.load(model_files[0])
            artifacts['model_name'] = os.path.basename(model_files[0]).replace('_model.joblib','')
        except Exception as e:
            errors.append(f"Error loading model: {e}")

    if not os.path.exists(scaler_file):
        errors.append("File scaler.joblib tidak ditemukan.")
    else:
        try:
            artifacts['scaler'] = joblib.load(scaler_file)
        except Exception as e:
            errors.append(f"Error loading scaler: {e}")

    if not os.path.exists(encoders_file):
        errors.append("File label_encoders.joblib tidak ditemukan.")
    else:
        try:
            artifacts['encoders'] = joblib.load(encoders_file)
        except Exception as e:
            errors.append(f"Error loading encoders: {e}")

    if not os.path.exists(columns_file):
        errors.append("File feature_columns.joblib tidak ditemukan.")
    else:
        try:
            artifacts['columns'] = joblib.load(columns_file)
        except Exception as e:
            errors.append(f"Error loading columns: {e}")

    return artifacts, errors

# --- Preprocess input ---
def preprocess_input(input_df, encoders, scaler, columns):
    df = input_df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            known_labels = list(le.classes_)
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in known_labels else -1)
            if -1 in df[col].values:
                st.warning(f"Value tidak dikenal pada kolom {col}, menggunakan encoding default.")
    for col in df.columns:
        if col not in encoders:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df = df[columns]
    scaled = scaler.transform(df)
    return scaled

# --- Load GPT2 model for recommendation ---
@st.cache_resource(show_spinner=False)
def load_gpt_model():
    model_name = "distilgpt2"  # bisa ganti ke model lain yang lebih kuat kalau ada resource
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_recommendation_gpt(input_text, tokenizer, model, max_length=100):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # Generate output (gunakan CPU/GPU jika ada)
    output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Buat output yang lebih bersih: hilangkan input_text dari output
    if output_text.startswith(input_text):
        output_text = output_text[len(input_text):].strip()
    return output_text

# --- Main app ---
def main_app():
    st.title("💼 Job Change Prediction + AI Recommendation")
    with st.sidebar:
        st.write(f"Hello, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

    artifacts, errors = load_artifacts()
    if errors:
        st.error("❌ Gagal load model atau artifacts:")
        for e in errors:
            st.write("- "+e)
        return

    model = artifacts['model']
    scaler = artifacts['scaler']
    encoders = artifacts['encoders']
    columns = artifacts['columns']

    # Input form
    city_opts = list(encoders['city'].classes_) if 'city' in encoders else ['city_103', 'city_40']
    gender_opts = list(encoders['gender'].classes_) if 'gender' in encoders else ['Male', 'Female']
    exp_opts = list(encoders['relevent_experience'].classes_) if 'relevent_experience' in encoders else ['Has relevent experience', 'No relevent experience']
    uni_opts = list(encoders['enrolled_university'].classes_) if 'enrolled_university' in encoders else ['no_enrollment', 'Part time course', 'Full time course']
    edu_opts = list(encoders['education_level'].classes_) if 'education_level' in encoders else ["Graduate", "Masters"]
    major_opts = list(encoders['major_discipline'].classes_) if 'major_discipline' in encoders else ["STEM", "Business Degree"]
    exp_total_opts = list(encoders['experience'].classes_) if 'experience' in encoders else ["<1", "1", "2"]
    size_opts = list(encoders['company_size'].classes_) if 'company_size' in encoders else ["<10", "10/49"]
    type_opts = list(encoders['company_type'].classes_) if 'company_type' in encoders else ["Pvt Ltd", "Funded Startup"]
    last_job_opts = list(encoders['last_new_job'].classes_) if 'last_new_job' in encoders else ["never", "1", "2"]

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.selectbox("City", city_opts)
            city_dev = st.slider("City Development Index", 0.0, 1.0, 0.7, 0.01)
            gender = st.selectbox("Gender", gender_opts)
        with col2:
            relevant_exp = st.selectbox("Relevant Experience", exp_opts)
            education_level = st.selectbox("Education Level", edu_opts)
            major = st.selectbox("Major Discipline", major_opts)
            experience = st.selectbox("Total Experience", exp_total_opts)
        with col3:
            enrolled_uni = st.selectbox("Enrolled University", uni_opts)
            company_size = st.selectbox("Company Size", size_opts)
            company_type = st.selectbox("Company Type", type_opts)
            last_new_job = st.selectbox("Years Since Last Job Change", last_job_opts)
            training_hours = st.number_input("Training Hours Completed", min_value=0, max_value=500, value=50)
        submit = st.form_submit_button("Predict & Recommend")

    if submit:
        input_df = pd.DataFrame({
            'city': [city],
            'city_development_index': [city_dev],
            'gender': [gender],
            'relevent_experience': [relevant_exp],
            'enrolled_university': [enrolled_uni],
            'education_level': [education_level],
            'major_discipline': [major],
            'experience': [experience],
            'company_size': [company_size],
            'company_type': [company_type],
            'last_new_job': [last_new_job],
            'training_hours': [training_hours]
        })

        try:
            processed = preprocess_input(input_df, encoders, scaler, columns)
            pred = model.predict(processed)[0]
            prob = model.predict_proba(processed)[0] if hasattr(model, 'predict_proba') else None

            st.subheader("Prediction Result:")
            if pred == 1:
                st.error("📈 Karyawan berpotensi *Resign*")
            else:
                st.success("🏢 Karyawan berpotensi *Stay*")

            if prob is not None:
                st.write(f"Confidence - Stay: {prob[0]:.1%}, Resign: {prob[1]:.1%}")

            # Load GPT model
            tokenizer, gpt_model = load_gpt_model()

            # Prepare prompt untuk generative AI
            prompt = f"Employee data:\nCity: {city}\nCity Development Index: {city_dev}\nGender: {gender}\nRelevant Experience: {relevant_exp}\nEducation Level: {education_level}\nMajor: {major}\nExperience: {experience}\nCompany Size: {company_size}\nCompany Type: {company_type}\nYears Since Last Job Change: {last_new_job}\nTraining Hours: {training_hours}\nPrediction: {'Resign' if pred==1 else 'Stay'}\n\nProvide advice on how to improve employee retention or why this employee might resign:"

            with st.spinner("Generating recommendation..."):
                recommendation = generate_recommendation_gpt(prompt, tokenizer, gpt_model, max_length=150)

            st.subheader("💡 AI Recommendation:")
            st.write(recommendation)

        except Exception as e:
            st.error(f"Error during prediction or recommendation: {e}")

# --- Main ---
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None

    if st.session_state.logged_in:
        main_app()
    else:
        login()

if __name__ == "__main__":
    main()