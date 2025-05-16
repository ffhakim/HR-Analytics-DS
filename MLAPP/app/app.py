import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="ML Predictor",
    page_icon="2️⃣",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 26px;
        color: #0D47A1;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# TITLE AND INTRODUCTION
# =====================================
st.markdown("<h1 class='main-header'>Machine Learning Model Evaluation & Prediction Platform</h1>", unsafe_allow_html=True)

with st.expander("ℹ️ About this app", expanded=False):
    st.markdown("""
    Dalam aplikasi ini anda dapat melakukan:
    1. Upload dataset training dan testing
    2. Melakukan reprocessing dan oversampling
    3. Train machine learning model
    4. Compare model performance dengan visualisasi
    5. Membuat prediksi untuk data baru
    6. Input custom value untuk melakukan prediksi
    
    **Models:**
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost
    """)

# =====================================
# LOAD DATA
# =====================================
st.markdown("<h2 class='sub-header'>📤 Upload Your Datasets</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
with col2:
    test_file = st.file_uploader("Upload Testing Data (CSV)", type=["csv"])

# inisialisasi state session
if 'models' not in st.session_state:
    st.session_state.models = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'le_dict' not in st.session_state:
    st.session_state.le_dict = {}
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'categorical_cols' not in st.session_state:
    st.session_state.categorical_cols = []
if 'numerical_cols' not in st.session_state:
    st.session_state.numerical_cols = []

if train_file and test_file:
    try:
        with st.spinner("Processing uploaded files..."):
            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)

            # data preview
            st.markdown("<h2 class='sub-header'>💾 Preview data</h2>", unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["Training Data", "Testing Data"])
            with tab1:
                st.dataframe(df_train.head())
                st.text(f"Training data shape: {df_train.shape}")
            with tab2:
                st.dataframe(df_test.head())
                st.text(f"Testing data shape: {df_test.shape}")

            # column names to lowercase to normalize 
            df_train.columns = df_train.columns.str.strip().str.lower()
            df_test.columns = df_test.columns.str.strip().str.lower()

            df_train['city_frequency'] = df_train.groupby('city')['city'].transform('count')
            df_test['city_frequency'] = df_test.groupby('city')['city'].transform('count')
            
            # validasi column 
            required_train_columns = {'target', 'enrollee_id', 'city_frequency'}
            if not required_train_columns.issubset(df_train.columns):
                st.error("Training file is missing required columns: target, enrollee_id, city_frequency")
                st.stop()
            
            if not {'enrollee_id', 'city_frequency'}.issubset(df_test.columns):
                st.error("Testing file is missing required columns: enrollee_id, city_frequency")
                st.stop()
            
            # =====================================
            # DATA EXPLORATION
            # =====================================
            st.markdown("<h2 class='sub-header'>🧭 Data Exploration</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(
                    names=['Will Not Resign (0)', 'Will Resign (1)'],
                    values=df_train['target'].value_counts().values,
                    title='Target Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # missing values chart
                missing_data = df_train.isnull().sum().sort_values(ascending=False)
                missing_data = missing_data[missing_data > 0]
                
                if not missing_data.empty:
                    fig = px.bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        title='Missing Values by Column',
                        labels={'x': 'Column', 'y': 'Missing Values Count'},
                        color_discrete_sequence=['#ff7043']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No missing values in the training data!")
            
            # =====================================
            # PREPROCESSING 
            # =====================================
            st.markdown("<h2 class='sub-header'>⚙️ Preprocessing</h2>", unsafe_allow_html=True)
            
            with st.spinner("Preprocessing data..."):
                le_dict = {}
                
                categorical_cols = []
                numerical_cols = []
                
                for col in df_train.columns:
                    if df_train[col].dtype == 'object':
                        df_train[col] = df_train[col].str.lower()
                        df_test[col] = df_test[col].str.lower()
                
                # handle missing values, encoding categorical values
                for col in df_train.columns:
                    if col not in ['target', 'enrollee_id', 'city_frequency']:
                        if df_train[col].dtype == 'object':
                            categorical_cols.append(col)
                            # fill categorical value with mode and encode
                            mode_val = df_train[col].mode()[0]
                            df_train[col] = df_train[col].fillna(mode_val)
                            df_test[col] = df_test[col].fillna(mode_val)
                            
                            # create and store label encoder
                            le = LabelEncoder()
                            df_train[col] = le.fit_transform(df_train[col])
                            if col in df_test.columns:
                                df_test[col] = le.transform(df_test[col])
                            le_dict[col] = le
                        else:
                            numerical_cols.append(col)
                            # mean for numerical value 
                            mean_val = df_train[col].mean()
                            df_train[col] = df_train[col].fillna(mean_val)
                            if col in df_test.columns:
                                df_test[col] = df_test[col].fillna(mean_val)
                
                # store in session state
                st.session_state.le_dict = le_dict
                st.session_state.categorical_cols = categorical_cols
                st.session_state.numerical_cols = numerical_cols
                
                # features and target
                X = df_train.drop(['target', 'enrollee_id', 'city_frequency'], axis=1)
                X_test = df_test.drop(['enrollee_id', 'city_frequency'], axis=1)
                y = df_train['target'].astype(float)
                
                # store feature names for custom prediction
                st.session_state.feature_names = X.columns.tolist()
                
                # scale numerical features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_test_scaled = scaler.transform(X_test)
                
                # store scaler in session state
                st.session_state.scaler = scaler
                
                # smote ykwhat
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X_scaled, y)

                # before and after smote comparison
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(
                        names=['Will Not Resign (0)', 'Will Resign (1)'],
                        values=[sum(y == 0), sum(y == 1)],
                        title='Before SMOTE',
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(
                        names=['Will Not Resign (0)', 'Will Resign (1)'],
                        values=[sum(y_res == 0), sum(y_res == 1)],
                        title='After SMOTE',
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # split data
                X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

                st.markdown("<div class='success-box'> preprocessing selesai </div>", unsafe_allow_html=True)
                
                # =====================================
                # TRAINING MODELS
                # =====================================
                st.markdown("<h2 class='sub-header'>Model Training and Evaluation</h2>", unsafe_allow_html=True)
                
                with st.spinner("Training models"):
                    models = {
                        'Logistic Regression': LogisticRegression(max_iter=2000),
                        'Decision Tree': DecisionTreeClassifier(),
                        'Random Forest': RandomForestClassifier(),
                        'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                    }
                    
                    # store models in session state
                    st.session_state.models = models

                    results = []
                    for name, model in models.items():
                        try:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_val)
                            results.append({
                                'Model': name,
                                'Accuracy': round(accuracy_score(y_val, y_pred), 4),
                                'Precision': round(precision_score(y_val, y_pred), 4),
                                'Recall': round(recall_score(y_val, y_pred), 4),
                                'F1-Score': round(f1_score(y_val, y_pred), 4)
                            })
                        except Exception as e:
                            st.error(f"Error training {name}: {e}")

                    results_df = pd.DataFrame(results)
                    if results_df.empty:
                        st.error("No models were successfully trained.")
                        st.stop()

                    # display model performance table
                    st.markdown("### Model Performance Comparison")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # model performance, visualize
                    fig = go.Figure()
                    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                    
                    for model_name in results_df['Model']:
                        model_results = results_df[results_df['Model'] == model_name]
                        fig.add_trace(go.Bar(
                            x=metrics,
                            y=[model_results[metric].values[0] for metric in metrics],
                            name=model_name
                        ))
                    
                    fig.update_layout(
                        title='Model Performance Comparison',
                        xaxis_title='Metric',
                        yaxis_title='Score',
                        barmode='group',
                        legend_title='Model'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # best model, which?
                    best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
                    st.session_state.best_model_name = best_model_name
                    best_model = models[best_model_name]
                    st.session_state.best_model = best_model
                    
                    st.markdown(f"<div class='success-box'>Best Model: <b>{best_model_name}</b> (F1-Score: {results_df.loc[results_df['F1-Score'].idxmax(), 'F1-Score']})</div>", unsafe_allow_html=True)
                    
                    # =====================================
                    # PREDIKSI DATA BARU
                    # =====================================
                    st.markdown("<h2 class='sub-header'>Prediction on Test Data</h2>", unsafe_allow_html=True)
                    
                    try:
                        new_predictions = best_model.predict(X_test_scaled)
                        new_prob = best_model.predict_proba(X_test_scaled)
                        
                        prediction_df = pd.DataFrame({
                            'Enrollee ID': df_test['enrollee_id'],
                            'Prediction': ['Will Resign' if pred == 1 else 'Will Not Resign' for pred in new_predictions],
                            'Resignation Probability': [round(prob[1], 4) for prob in new_prob]
                        })
                        
                        prediction_df['Risk Level'] = prediction_df['Resignation Probability'].apply(
                            lambda x: 'High' if x >= 0.7 else 'Medium' if x >= 0.4 else 'Low'
                        )
                        
                        st.dataframe(prediction_df, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.pie(
                                names=['Will Not Resign', 'Will Resign'],
                                values=[(new_predictions == 0).sum(), (new_predictions == 1).sum()],
                                title='Prediction Distribution',
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.histogram(
                                prediction_df,
                                x='Resignation Probability',
                                color='Risk Level',
                                title='Resignation Probability Distribution',
                                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # download predictions
                        csv = prediction_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="resignation_predictions.csv",
                            mime="text/csv",
                            key='download-csv'
                        )
                        
                    except Exception as e:
                        st.error(f"Error in prediction: {e}")
                        st.stop()

    except Exception as e:
        st.error(f"Failed to process files: {e}")
        st.stop()




# CUSTOM PREDICTION USING INPUT

if st.session_state.best_model is not None:
    st.markdown("<h2 class='sub-header'>Custom Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Masukan value untuk melakukan prediksi</div>", unsafe_allow_html=True)
    
    with st.form(key='custom_prediction_form'):
        st.subheader("Enter Employee Details")
        
        cols = st.columns(3)
        input_values = {}
        
        col_index = 0
        for feature in st.session_state.feature_names:
            with cols[col_index % 3]:
                if feature in st.session_state.categorical_cols:
                    possible_values = st.session_state.le_dict[feature].classes_
                    input_values[feature] = st.selectbox(
                        f"{feature.replace('_', ' ').title()}",
                        options=possible_values
                    )
                else:
                    input_values[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}",
                        value=0.0,
                        step=0.1
                    )
            col_index += 1
        
        submit_button = st.form_submit_button(label="Predict")
        
        
    if submit_button:
        try:
            input_df = pd.DataFrame([input_values])
            
            for feature in st.session_state.categorical_cols:


                if feature in input_df.columns and feature in st.session_state.le_dict:
                    le = st.session_state.le_dict[feature]


                    try:
                        input_df[feature] = le.transform([input_df[feature].values[0]])
                    except ValueError:
                        st.warning(f"no category for {feature}. using most common category instead (mode)")
                        input_df[feature] = 0  # default return to first category 
            


            input_scaled = st.session_state.scaler.transform(input_df)
            prediction = st.session_state.best_model.predict(input_scaled)[0]
            probability = st.session_state.best_model.predict_proba(input_scaled)[0][1]
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("### Prediksi: Akan Resign")
                else:
                    st.success("### Prediksi: Tidak akan Resign")
            with col2:
                st.metric("Kemungkinan Resign", f"{probability:.2%}")
            


            risk_level = "Tinggi" if probability >= 0.7 else "Medium" if probability >= 0.4 else "Rendah"
            risk_color = "red" if risk_level == "Tinggi" else "orange" if risk_level == "Medium" else "green"
            
            st.markdown(f"<h3 style='color: {risk_color}'>Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
            
            if risk_level == "Tinggi":
                st.markdown("""
                <div class='warning-box'>
                <h4>Recommendation</h4>
                <p>Karyawan ini memiliki kemungkinan pindah kerja tinggi lakukan:</p>
                <ul>
                    <li>One on one meeting</li>
                    <li>Review benefit</li>
                    <li>Membahas kemungkinan kenaikan jabatan atau karir yang baik</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            if hasattr(st.session_state.best_model, 'feature_importances_'):
                st.markdown("### Feature Importance")
                
                importances = st.session_state.best_model.feature_importances_
                feature_imp = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                
                fig = px.bar(
                    feature_imp,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance',
                    color_discrete_sequence=['#1E88E5']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("upload dataset untuk melakukan prediksi")

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; color: #888;">
    <p>FinPro Group-2</p>
</div>
""", unsafe_allow_html=True)