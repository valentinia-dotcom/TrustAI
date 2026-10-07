import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

st.set_page_config(page_title='TrustAI', page_icon='🛡️', layout='wide')
st.title('🛡️ TrustAI')
st.caption('A Web-Based Explainable Platform for Trustworthiness Assessment and Improvement of Machine Learning Models')
defaults = {
    'data': None, 'test': None, 'model': None, 'model2': None, 'target': None, 'task': None,
    'performance': None, 'data_quality': None, 'stability': None, 'explainability': None, 'trust_score': None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource
def load_model(file):
    return joblib.load(file)

def prepare_data(df, target, model):
    X = df.drop(columns=[target])
    y = df[target]
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
        missing = [x for x in features if x not in X.columns]
        if missing:
            raise ValueError(f'Missing model features: {missing}')
        X = X[features]
    return X, y

def evaluate(model, X, y, task):
    pred = model.predict(X)
    if task == 'Classification':
        acc = accuracy_score(y, pred)
        pre = precision_score(y, pred, average='weighted', zero_division=0)
        rec = recall_score(y, pred, average='weighted', zero_division=0)
        f1 = f1_score(y, pred, average='weighted', zero_division=0)
        score = np.mean([acc, pre, rec, f1]) * 100
        return {'Accuracy': acc, 'Precision': pre, 'Recall': rec, 'F1': f1, 'Score': score}
    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    target_range = float(y.max() - y.min())
    nrmse = rmse / target_range if target_range else 0
    score = (0.5 * np.clip(r2, 0, 1) + 0.5 * np.clip(1 - nrmse, 0, 1)) * 100
    return {'R²': r2, 'MAE': mae, 'RMSE': rmse, 'Score': score}

def data_quality(df):
    if df.empty:
        return 0
    missing = df.isna().sum().sum() / df.size
    duplicates = df.duplicated().mean()
    numeric = df.select_dtypes(include=np.number)
    if numeric.empty:
        infinite = 0
    else:
        infinite = np.isinf(numeric.to_numpy()).mean()
    penalty = missing * 50 + duplicates * 30 + infinite * 20
    return float(np.clip(100 - penalty, 0, 100))

def stability(model, X, y, task):
    if len(X) < 10:
        return 50.0
    scores = []
    for seed in range(5):
        sample = X.sample(frac=0.8, random_state=seed)
        ys = y.loc[sample.index]
        try:
            result = evaluate(model, sample, ys, task)
            scores.append(result['Score'])
        except Exception:
            pass
    if len(scores) < 2:
        return 50.0
    variation = np.std(scores)
    return float(np.clip(100 - variation * 5, 0, 100))

def risk(score):
    if score >= 80:
        return 'Low'
    if score >= 60:
        return 'Medium'
    return 'High'

def trust_score(performance, quality, stability, explainability):
    return 0.40 * performance + 0.20 * quality + 0.20 * stability + 0.20 * explainability

page = st.sidebar.radio('Navigation', [
    '🏠 Overview', '⚙️ Upload & Setup', '📊 Evaluate', '🔍 Explain',
    '🛡️ Trust Score', '🔄 Model Comparison', '💡 Recommendations'
])
if page == '🏠 Overview':
    st.header('TrustAI')
    st.write("""
        TrustAI evaluates trained machine learning models beyond
        predictive accuracy.

        The platform combines model evaluation, data quality,
        stability and explainability to provide a structured
        Trust Indicator and improvement recommendations.
        """)
    st.subheader('Assessment Workflow')
    st.info('Upload → Evaluate → Explain with SHAP & LIME → Trust Score → Compare Models → Recommend Improvements')
    st.warning('The Trust Score is a configurable model-audit indicator. It is not a certified or universal measure of AI trustworthiness.')
elif page == '⚙️ Upload & Setup':
    st.header('⚙️ Upload Dataset & Model')
    data_file = st.file_uploader('Dataset (CSV)', type='csv', key='data_upload')
    if data_file:
        st.session_state.data = pd.read_csv(data_file)
    if st.session_state.data is not None:
        df = st.session_state.data
        st.success(f'Dataset loaded: {len(df)} rows × {len(df.columns)} columns')
        st.dataframe(df.head(), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.target = st.selectbox('Target column', df.columns, key='target_select')
        with c2:
            st.session_state.task = st.selectbox('Task type', ['Classification', 'Regression'], key='task_select')
        test_file = st.file_uploader('Held-out evaluation dataset (optional)', type='csv', key='test_upload')
        if test_file:
            st.session_state.test = pd.read_csv(test_file)
    model_file = st.file_uploader('Primary model (.pkl / .joblib)', type=['pkl', 'joblib'], key='model_upload')
    if model_file:
        try:
            st.session_state.model = load_model(model_file)
            st.success(f'Primary model loaded: {type(st.session_state.model).__name__}')
        except Exception as e:
            st.error(f'Model could not be loaded: {e}')
    model2_file = st.file_uploader('Comparison model (optional)', type=['pkl', 'joblib'], key='model2_upload')
    if model2_file:
        try:
            st.session_state.model2 = load_model(model2_file)
            st.success(f'Comparison model loaded: {type(st.session_state.model2).__name__}')
        except Exception as e:
            st.error(f'Comparison model could not be loaded: {e}')
    if st.session_state.data is not None and st.session_state.model is not None and st.session_state.target:
        st.success('Setup complete. Continue to Evaluate.')
elif page == '📊 Evaluate':
    st.header('📊 Model Evaluation')
    if st.session_state.data is None or st.session_state.model is None or not st.session_state.target:
        st.warning('Complete Upload & Setup first.')
        st.stop()
    df = st.session_state.test if st.session_state.test is not None else st.session_state.data
    if st.session_state.test is None:
        st.warning('No held-out dataset was uploaded. Evaluation is using the main dataset.')
    else:
        st.success('Using held-out evaluation data.')
    try:
        X, y = prepare_data(df, st.session_state.target, st.session_state.model)
        results = evaluate(st.session_state.model, X, y, st.session_state.task)
        performance = results['Score']
        quality = data_quality(df)
        stable = stability(st.session_state.model, X, y, st.session_state.task)
        st.session_state.performance = performance
        st.session_state.data_quality = quality
        st.session_state.stability = stable
        st.subheader('Performance')
        cols = st.columns(len(results) - 1)
        metrics = [(k, v) for k, v in results.items() if k != 'Score']
        for col, (name, value) in zip(cols, metrics):
            col.metric(name, f'{value:.3f}')
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric('Performance Score', f'{performance:.1f}/100')
        c2.metric('Data Quality', f'{quality:.1f}/100')
        c3.metric('Stability', f'{stable:.1f}/100')
        st.write(f'**Performance Risk:** {risk(performance)}')
        st.caption('These scores are configurable diagnostic indicators intended to support model review.')
    except Exception as e:
        st.error(f'Evaluation failed: {e}')
elif page == '🔍 Explain':
    st.header('🔍 Model Explainability')
    if st.session_state.data is None or st.session_state.model is None or not st.session_state.target:
        st.warning('Complete Upload & Setup first.')
        st.stop()
    df = st.session_state.test if st.session_state.test is not None else st.session_state.data
    model = st.session_state.model
    try:
        X, y = prepare_data(df, st.session_state.target, model)
        X_sample = X.sample(min(100, len(X)), random_state=42)
        st.subheader('Feature Importance')
        result = permutation_importance(model, X_sample, y.loc[X_sample.index], n_repeats=3, random_state=42)
        importance = pd.DataFrame({'Feature': X_sample.columns, 'Importance': result.importances_mean}).sort_values(
            'Importance', ascending=False)
        st.bar_chart(importance.set_index('Feature').head(10))
        st.subheader('SHAP Explanation')
        shap_success = False
        try:
            background = X_sample.iloc[:min(30, len(X_sample))]
            explainer = shap.Explainer(model.predict, background)
            shap_values = explainer(X_sample.iloc[:min(20, len(X_sample))])
            values = np.abs(shap_values.values)
            if values.ndim > 2:
                values = values.mean(axis=-1)
            shap_importance = values.mean(axis=0)
            shap_df = pd.DataFrame({'Feature': X_sample.columns, 'SHAP Importance': shap_importance}).sort_values(
                'SHAP Importance', ascending=False)
            st.bar_chart(shap_df.set_index('Feature').head(10))
            st.success('SHAP explanation generated successfully.')
            shap_success = True
        except Exception as e:
            st.warning(f'SHAP could not explain this model automatically: {e}')
        st.subheader('LIME Local Explanation')
        lime_success = False
        try:
            numeric_X = X_sample.select_dtypes(include=np.number)
            if len(numeric_X.columns) != len(X_sample.columns):
                raise ValueError('LIME requires numeric model input in this compact implementation.')
            mode = 'classification' if st.session_state.task == 'Classification' else 'regression'
            lime = LimeTabularExplainer(X_sample.to_numpy(), feature_names=list(X_sample.columns), mode=mode, random_state=42)
            row = X_sample.iloc[0]
            if mode == 'classification':
                if not hasattr(model, 'predict_proba'):
                    raise ValueError('The model does not provide predict_proba().')
                explanation = lime.explain_instance(row.to_numpy(), model.predict_proba, num_features=min(10, X.shape[1]))
            else:
                explanation = lime.explain_instance(row.to_numpy(), model.predict, num_features=min(10, X.shape[1]))
            lime_df = pd.DataFrame(explanation.as_list(), columns=['Feature', 'Contribution'])
            st.dataframe(lime_df, use_container_width=True, hide_index=True)
            st.success('LIME local explanation generated successfully.')
            lime_success = True
        except Exception as e:
            st.warning(f'LIME could not explain this model automatically: {e}')
        methods = [True, shap_success, lime_success]
        explainability = sum(methods) / len(methods) * 100
        st.session_state.explainability = explainability
        st.metric('Explainability Score', f'{explainability:.1f}/100')
        st.caption('The Explainability Score reflects whether the configured explanation methods were successfully generated. '
                   'It does not measure explanation correctness.')
    except Exception as e:
        st.error(f'Explainability analysis failed: {e}')
elif page == '🛡️ Trust Score':
    st.header('🛡️ Trust Score')
    required = [st.session_state.performance, st.session_state.data_quality, st.session_state.stability, st.session_state.explainability]
    if any(x is None for x in required):
        st.warning('Complete Evaluate and Explain before calculating the Trust Score.')
        st.stop()
    score = trust_score(st.session_state.performance, st.session_state.data_quality,
                        st.session_state.stability, st.session_state.explainability)
    st.session_state.trust_score = score
    st.metric('Overall Trust Indicator', f'{score:.1f}/100')
    st.metric('Overall Risk', risk(score))
    components = pd.DataFrame({
        'Component': ['Performance', 'Data Quality', 'Stability', 'Explainability'],
        'Score': [st.session_state.performance, st.session_state.data_quality,
                  st.session_state.stability, st.session_state.explainability],
        'Weight': ['40%', '20%', '20%', '20%']
    })
    st.dataframe(components, use_container_width=True, hide_index=True)
    st.bar_chart(components.set_index('Component')['Score'])
    st.warning('The Trust Score is a configurable model-audit indicator for comparative review. '
               'It is not a universal, regulatory, or certified measure of model trustworthiness.')
elif page == '🔄 Model Comparison':
    st.header('🔄 Model Comparison')
    if st.session_state.model2 is None:
        st.info('Upload a comparison model in Upload & Setup.')
        st.stop()
    if st.session_state.data is None or st.session_state.model is None:
        st.warning('Complete setup first.')
        st.stop()
    df = st.session_state.test if st.session_state.test is not None else st.session_state.data
    try:
        X1, y = prepare_data(df, st.session_state.target, st.session_state.model)
        X2, _ = prepare_data(df, st.session_state.target, st.session_state.model2)
        primary = evaluate(st.session_state.model, X1, y, st.session_state.task)
        comparison = evaluate(st.session_state.model2, X2, y, st.session_state.task)
        table = pd.DataFrame({'Model': ['Primary Model', 'Comparison Model'], 'Performance Score': [primary['Score'], comparison['Score']]})
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.bar_chart(table.set_index('Model'))
        if primary['Score'] > comparison['Score']:
            st.success('Primary model has the stronger predictive performance on this evaluation dataset.')
        elif comparison['Score'] > primary['Score']:
            st.success('Comparison model has the stronger predictive performance on this evaluation dataset.')
        else:
            st.info('Both models have the same Performance Score.')
        st.caption('This comparison evaluates predictive performance. '
                   'Trust-related dimensions should also be reviewed before selecting a model.')
    except Exception as e:
        st.error(f'Model comparison failed: {e}')
elif page == '💡 Recommendations':
    st.header('💡 Recommended Improvements')
    if st.session_state.trust_score is None:
        st.warning('Complete Evaluate, Explain and Trust Score first.')
        st.stop()
    recommendations = []
    if st.session_state.performance < 80:
        recommendations.append('Improve predictive performance through feature engineering, model tuning, or alternative algorithms.')
    if st.session_state.data_quality < 80:
        recommendations.append('Improve data quality by reviewing missing values, duplicate records and invalid numeric values.')
    if st.session_state.stability < 80:
        recommendations.append('Investigate model stability using additional validation samples or cross-validation.')
    if st.session_state.explainability < 80:
        recommendations.append("Review SHAP and LIME compatibility and improve the model's explanation pipeline.")
    if st.session_state.trust_score < 60:
        recommendations.append('Avoid high-impact deployment until the identified model risks have been reviewed.')
    st.metric('Current Trust Indicator', f'{st.session_state.trust_score:.1f}/100')
    st.write(f'**Risk Level:** {risk(st.session_state.trust_score)}')
    if recommendations:
        for i, recommendation in enumerate(recommendations, start=1):
            st.write(f'**{i}.** {recommendation}')
    else:
        st.success('No major weaknesses were identified by the configured TrustAI indicators. '
                   'Continue monitoring the model with new and representative data.')
    st.divider()
    st.caption('Recommendations are generated from configurable diagnostic thresholds and should support, not replace, expert review.')
