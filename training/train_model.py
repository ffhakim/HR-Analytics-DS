import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import warnings
import json
import os
import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FILE = 'data/raw/aug_train.csv'
TARGET_COLUMN = 'target'
ID_COLUMN = 'enrollee_id'
ARTIFACT_DIR = 'models/'
MODEL_PERF_FILE = 'model_performance.json'
MODEL_VERSION = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create directories
Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)

# --- Load Data ---
print(f"🔍 Loading data from {DATA_FILE}...")
try:
    df_train = pd.read_csv(DATA_FILE)
    print(f"✅ Loaded {len(df_train)} records with {len(df_train.columns)} features")
except FileNotFoundError:
    print(f"❌ ERROR: Data file '{DATA_FILE}' not found")
    exit()

# --- Preprocessing ---
print("\n⚙️ Preprocessing data...")
df_train = df_train.drop(ID_COLUMN, axis=1)

# Handle missing values
for col in df_train.select_dtypes(include='object').columns:
    mode_val = df_train[col].mode()[0]
    df_train[col].fillna(mode_val, inplace=True)
    print(f"  - {col}: Filled {df_train[col].isna().sum()} missing values with mode '{mode_val}'")

for col in df_train.select_dtypes(include=['int64', 'float64']).columns:
    if col != TARGET_COLUMN:
        median_val = df_train[col].median()
        df_train[col].fillna(median_val, inplace=True)
        print(f"  - {col}: Filled {df_train[col].isna().sum()} missing values with median {median_val:.2f}")

# Encode categorical features
categorical_cols = df_train.select_dtypes(include='object').columns.tolist()
numerical_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove(TARGET_COLUMN)

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    encoders[col] = le
    print(f"  - Encoded '{col}' with {len(le.classes_)} categories")

# Prepare features and target
X = df_train.drop(TARGET_COLUMN, axis=1)
y = df_train[TARGET_COLUMN]
print(f"\nℹ️ Class distribution: {y.value_counts().to_dict()}")

# Save preprocessing artifacts
joblib.dump(encoders, f'{ARTIFACT_DIR}label_encoders.joblib')
joblib.dump(X.columns.tolist(), f'{ARTIFACT_DIR}feature_columns.joblib')
print("\n💾 Saved preprocessing artifacts")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, f'{ARTIFACT_DIR}scaler.joblib')

# Handle class imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)
print(f"⚖️ Applied SMOTE - new class distribution: {pd.Series(y_res).value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
print(f"\n📊 Dataset split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# --- Model Training ---
print("\n🧠 Training models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

# Hyperparameter grids for tuning
param_grids = {
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [5, 10]},
    'XGBoost': {'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
}

results = []
best_model = None
best_model_name = None
best_f1_score = 0

for name, model in models.items():
    print(f"\n🔧 Training {name}...")
    
    # Hyperparameter tuning for selected models
    if name in param_grids:
        grid = GridSearchCV(model, param_grids[name], cv=3, scoring='f1')
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"  - Best params: {grid.best_params_}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1-Score': round(f1_score(y_test, y_pred), 4),
        'ROC AUC': round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None
    }
    results.append(metrics)
    
    f1 = metrics['F1-Score']
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = model
        best_model_name = name

# Save best model
if best_model:
    model_filename = f'{ARTIFACT_DIR}{best_model_name.replace(" ", "_")}_model.joblib'
    joblib.dump(best_model, model_filename)
    print(f"\n🏆 Best model: {best_model_name} (F1: {best_f1_score:.4f})")
    print(f"💾 Saved as {model_filename}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(f'{ARTIFACT_DIR}model_comparison.csv', index=False)
with open(MODEL_PERF_FILE, 'w') as f:
    json.dump({
        'best_model': best_model_name,
        'best_f1': best_f1_score,
        'model_version': MODEL_VERSION,
        'metrics': results
    }, f, indent=2)

print("\n📊 Model Comparison Results:")
print(results_df[['Model', 'F1-Score', 'ROC AUC', 'Accuracy']])
print("\n✅ Training process completed successfully!")