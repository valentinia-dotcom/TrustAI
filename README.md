\# TrustAI



\## A Web-Based Explainable Platform for Trustworthiness Assessment of Machine Learning Models



TrustAI is a Streamlit-based application developed to evaluate and explain machine learning models using multiple trust-related factors instead of relying only on model accuracy.



The platform supports both classification and regression models and combines model performance, data quality, stability, explainability, and group-level fairness assessment into an overall Trust Score.



\## Key Features



\- Upload CSV datasets

\- Upload trained machine learning models

\- Classification and regression support

\- Held-out evaluation dataset support

\- Model performance evaluation

\- SHAP-based explainability

\- LIME-based local explanations

\- Data quality assessment

\- Stability assessment across data segments

\- Group-level fairness assessment

\- Explainability Coverage Score

\- Overall Trust Score

\- Multi-model comparison

\- Trust-based recommendations



\## Trustworthiness Assessment



TrustAI evaluates a model using five components:



\### 1. Performance

For classification models, metrics such as Accuracy, Precision, Recall, F1 Score and ROC-AUC are evaluated.



For regression models, MAE, MSE, RMSE and R² are evaluated.



\### 2. Data Quality

The application checks the evaluation dataset for:



\- Missing values

\- Duplicate rows



These checks are combined into a Data Quality Score.



\### 3. Stability

The evaluation data is divided into multiple segments and model performance is measured across them.



Variation in performance is used as an indicator of model stability.



\### 4. Explainability

TrustAI uses two explainable AI techniques:



\- SHAP (SHapley Additive exPlanations)

\- LIME (Local Interpretable Model-agnostic Explanations)



The Explainability Coverage Score indicates whether these explanation methods were successfully produced.



\### 5. Group-Level Fairness

Users can select an attribute and compare model performance across different groups.



For classification, group accuracy differences are evaluated.



For regression, group error differences are evaluated.



This is a prototype group-performance assessment and should not be interpreted as a complete fairness certification.



\## Overall Trust Score



The Overall Trust Score combines:



\- Performance Score

\- Data Quality Score

\- Stability Score

\- Explainability Coverage Score

\- Fairness Score, when evaluated



The Trust Score is a project-defined assessment indicator designed to support model review. It is not a formal certification of model trustworthiness.



\## Multi-Model Comparison



TrustAI allows two compatible models for the same machine learning task to be evaluated on the same dataset.



The comparison includes:



\- Performance

\- Data Quality

\- Stability

\- Explainability Coverage

\- Fairness

\- Overall Trust Score



This helps identify situations where model performance and broader trust-related assessment may lead to different model rankings.



\## Technologies Used



\- Python

\- Streamlit

\- Pandas

\- NumPy

\- Scikit-learn

\- SHAP

\- LIME

\- Matplotlib

\- Joblib



\## Installation



Clone the repository:



```bash

git clone https://github.com/valentinia-dotcom/TrustAI.git

cd TrustAI

