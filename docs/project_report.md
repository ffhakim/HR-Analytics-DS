# HR Analytics Project Report

## 1. Project Overview
Predict employee retention risk for data scientists using ML and AI

## 2. Dataset
- Source: Kaggle HR Analytics Dataset
- Size: 15,000 records
- Features: 12 employee attributes
- Target: `target` (0 = stay, 1 = leave)

## 3. Methodology
### Preprocessing
- Missing value imputation
- Categorical encoding
- Feature scaling
- SMOTE for class imbalance

### Model Training
- Algorithms tested: Logistic Regression, Decision Tree, Random Forest, XGBoost
- Best model: Random Forest (F1: 0.85)
- Hyperparameter tuning with GridSearchCV

### AI Recommendations
- DistilGPT-2 model
- Prompt engineering for retention strategies


## 4. Conclusion
The model achieves 85% accuracy with actionable recommendations