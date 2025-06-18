# HR Analytics: Job Change Prediction for Data Scientists

![App Screenshot](app/assets/hr_analytics.png)

Predict employee retention risks and get AI-powered recommendations for data science professionals.

## Features
- 🧠 Machine learning-powered attrition prediction
- 🤖 GPT-2 powered retention recommendations
- 🔐 Role-based authentication
- 📊 Interactive employee profile analysis
- 📈 Probability confidence scores

## Installation
```bash
# Clone repository
git clone https://github.com/HR-Analytics-DS.git
cd HR-Analytics-DS

# Install dependencies
# For model training environment
pip install -r training/requirements.txt

# For Streamlit application
pip install -r app/requirements.txt

# Run Streamlit app
streamlit run app/streamlit_app.py
```

## Usage
1. Login with demo credentials:
   - Admin: `admin` / `user123`
   - User: `user` / `password123`
2. Fill in employee details
3. Get retention prediction
4. Review AI-powered recommendations

## Project Structure
```
HR-Analytics-DS/
├── data/                           # Data files
│   └── raw/                        # Raw datasets
│       └── aug_train.csv           # Training data
├── models/                         # Trained models and artifacts
│   ├── Random_Forest_model.joblib  # Best model
│   ├── scaler.joblib               # Feature scaler
│   ├── label_encoders.joblib       # Categorical encoders
│   ├── feature_columns.joblib      # Feature columns
│   └── model_comparison.csv        # Model evaluation results
├── app/                            # Streamlit application
│   ├── streamlit_app.py            # Main application
│   └── requirements.txt            # App dependencies
├── training/                       # Model training scripts
│   ├── train_model.py              # Training pipeline
│   └── requirements.txt            # Training dependencies
├── docs/                           # Documentation
│   ├── project_report.md           # Technical documentation
│   └── model_card.md               # Model documentation
├── tests/                          # Test scripts
├── .gitignore                      # Files to ignore in Git
├── LICENSE                         # Project license
├── README.md                       # Project overview
└── model_performance.json          # Model metrics
```

## 🧠 Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.81     | 0.78      | 0.74   | 0.76     | 0.87    |
| Decision Tree       | 0.79     | 0.75      | 0.72   | 0.73     | 0.78    |
| **Random Forest**   | **0.85** | **0.82**  | **0.79**| **0.80** | **0.89**|
| XGBoost             | 0.83     | 0.80      | 0.77   | 0.78     | 0.88    |

*Random Forest was selected as the best performing model*

## Contributors
- [Affan Moshe](https://github.com/affanmoshe)
- [Denina Nastiti Putri Amani](https://github.com/deninanastiti)
- [Mohammad Fahmi Hakim](https://github.com/ffhakim)
- [Mukhlizardy Al Fauzan](https://github.com/Mukhlizardy)
- [Zahra Vony](https://github.com/zahravony507)

## License
This project is licensed under the MIT License - see [LICENSE](https://github.com/ffhakim/HR-Analytics-DS/blob/main/LICENSE.txt) for details.
