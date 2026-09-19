import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, mean_absolute_error,mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import shap
from lime.lime_tabular import LimeTabularExplainer 
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

@st.cache_resource
def load_model_cached(model_file):
    return joblib.load(model_file)

st.title("TrustAI")
st.subheader(
    "A Web-Based Explainable Platform for Trustworthiness "
    "Assessment and Improvement of Machine Learning Models")

st.write(
    "TrustAI helps evaluate, explain, and assess the "
    "trustworthiness of machine learning models.")

# -----------------------------
# STEP 1: UPLOAD DATASET
# -----------------------------

st.header("1. Upload Dataset")

data_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    key="dataset_uploader")

if data_file is not None:
    df = pd.read_csv(data_file)
    st.success("Dataset loaded successfully!")
    st.write("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # STEP 2: SELECT TARGET
    # -----------------------------
    st.header("2. Select Target Column")
    target_column = st.selectbox(
        "Select the column you want to predict",
        df.columns)
    task_type = st.selectbox("Select Machine Learning Task",["Classification", "Regression"])
    # -----------------------------
    # HELD-OUT EVALUATION DATASET
    # -----------------------------

    st.subheader("Held-Out Evaluation Dataset")
    test_file = st.file_uploader("Upload held-out test CSV", type=["csv"], key="heldout_dataset_uploader")

    if test_file is not None:

        evaluation_df = pd.read_csv(test_file)

        if target_column not in evaluation_df.columns:
            st.error("Held-out dataset must contain the target column: "
            + target_column)
            st.stop()

        st.success("Held-out evaluation dataset loaded successfully!")

        st.caption("Model evaluation and TrustAI assessment will use "
 "this held-out dataset.")

    else:

        evaluation_df = df

        st.warning("No held-out dataset uploaded. The original uploaded "
        "dataset will be used for evaluation. Do not describe "
        "these results as held-out performance.")

    X = evaluation_df.drop(columns=[target_column])
    y = evaluation_df[target_column]

    st.write("Evaluation Rows:", len(evaluation_df))
    st.write("Features:", X.shape[1])
    st.write("Target:", target_column)
    st.write("Task Type:", task_type)
    
    # -----------------------------
    # STEP 3: UPLOAD MODEL
    # -----------------------------

    st.header("3. Upload Trained Model")
    model_file = st.file_uploader( "Choose a trained model", type=["pkl", "joblib"], key="model_uploader")
    comparison_model_file = st.file_uploader(
        "Choose a comparison model (optional)",
        type=["pkl", "joblib"],
        key="comparison_model_uploader")

    if model_file is not None:
        try:
            model = load_model_cached(model_file)
            st.success("Model loaded successfully!")
            predictions = model.predict(X)
            st.success("Dataset is compatible with the model!")
            st.write("First 5 Predictions:")
            st.write(predictions[:5])

# -----------------------------
# COMPARISON MODEL
# -----------------------------

            comparison_model = None
            comparison_predictions = None

            if comparison_model_file is not None:
                comparison_model = load_model_cached(comparison_model_file)
                comparison_predictions = comparison_model.predict(X)

                st.success("Comparison model loaded successfully!")

            st.header("4. Model Evaluation")
            if task_type == "Classification":
                st.subheader("Classification Metrics")
                accuracy = accuracy_score(y, predictions)
                precision = precision_score(y, predictions)
                recall = recall_score(y, predictions)
                f1 = f1_score(y, predictions)
                probabilities = model.predict_proba(X)[:, 1]
                roc_auc = roc_auc_score(y, probabilities)

                st.write("Accuracy:", round(accuracy, 4))
                st.write("Precision:", round(precision, 4))
                st.write("Recall:", round(recall, 4))
                st.write("F1 Score:", round(f1, 4))
                st.write("ROC-AUC:", round(roc_auc, 4))
                
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y, predictions,labels=model.classes_)
                class_labels = model.classes_

                cm_df = pd.DataFrame( cm, index=[f"Actual {label}" for label in class_labels], columns=[f"Predicted {label}" for label in class_labels])

                st.dataframe(cm_df)
            else:
                st.subheader("Regression Metrics")
                mae = mean_absolute_error(y, predictions)
                mse = mean_squared_error(y, predictions)
                rmse = mse ** 0.5
                r2 = r2_score(y, predictions)
                st.write("MAE:", round(mae, 4))
                st.write("MSE:", round(mse, 4))
                st.write("RMSE:", round(rmse, 4))
                st.write("R² Score:", round(r2, 4))
            if comparison_model is not None and task_type == "Regression":

                comparison_mae = mean_absolute_error( y, comparison_predictions)
                comparison_mse = mean_squared_error( y, comparison_predictions )

                comparison_rmse = comparison_mse ** 0.5

                comparison_r2 = r2_score( y, comparison_predictions)

                st.subheader("Multi-Model Performance Comparison")

                primary_name = type( model.named_steps["model"] ).__name__

                comparison_name = type( comparison_model.named_steps["model"] ).__name__

                comparison_df = pd.DataFrame({ "Metric": ["MAE", "RMSE", "R²"], primary_name: [ round(mae, 2), round(rmse, 2), round(r2, 4) ],
                    comparison_name: [ round(comparison_mae, 2), round(comparison_rmse, 2), round(comparison_r2, 4) ] })

                st.dataframe( comparison_df, hide_index=True)

            st.header("5. Explainability")

            permutation_available = False
            shap_available = False  
            lime_available = False
            st.write("Calculating feature importance...")
            importance_result = permutation_importance(model,X,y,n_repeats=5,random_state=42)
            permutation_available = True
            importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance_result.importances_mean})
            importance_df = importance_df.sort_values(by="Importance",ascending=False)
            st.write("Feature Importance")
            st.dataframe(importance_df)
            st.subheader("Top 10 Important Features")
            top_features = importance_df.head(10)
            chart_data = top_features.set_index("Feature")["Importance"]
            st.bar_chart(chart_data)
            st.subheader("Model Structure")
            st.write("Model Type:", type(model).__name__)
            if hasattr(model, "named_steps"):
                st.write("Pipeline Steps:")
                st.write(list(model.named_steps.keys()))
                final_model = model.named_steps["model"]
                st.write("Final ML Model:", type(final_model).__name__)
                preprocessor = model.named_steps["preprocessor"]
                X_transformed = preprocessor.transform(X)
                st.write("Original Features:", X.shape[1])
                st.write("Features After Preprocessing:", X_transformed.shape[1])
                feature_names = preprocessor.get_feature_names_out()
                st.write("Features Used by Model:")
                st.write(feature_names)

                st.subheader("SHAP Explainability")

                # Use a smaller sample so SHAP runs faster
                X_shap = X_transformed[:100]

                if type(final_model).__name__ in ["RandomForestRegressor", "RandomForestClassifier", "DecisionTreeRegressor", "DecisionTreeClassifier" ]:
                    explainer = shap.TreeExplainer(final_model)
                    shap_values = explainer.shap_values(X_shap)

                elif type(final_model).__name__ in [ "LinearRegression", "LogisticRegression"]:
                    explainer = shap.LinearExplainer( final_model, X_transformed )
                    shap_values = explainer.shap_values(X_shap)

                else:
                    st.warning( "SHAP explanation is not currently supported "
        "for this model type.")
                    shap_values = None

                if shap_values is not None:
                    shap_available = True
                    st.success("SHAP values calculated successfully!")
                    st.subheader("SHAP Summary Plot")

                    plt.figure()

                    if task_type == "Classification":
                        if isinstance(shap_values, list):
                            plot_shap_values = shap_values[1]
                        elif np.asarray(shap_values).ndim == 3:
                            plot_shap_values = shap_values[:, :, 1]
                        else:
                            plot_shap_values = shap_values

                        shap.summary_plot( plot_shap_values, X_shap, feature_names=feature_names, show=False )
                    else:
                        shap.summary_plot( shap_values, X_shap, feature_names=feature_names, show=False)
                    st.pyplot(plt.gcf())
                    plt.close()                

                st.subheader("LIME Explanation")
                # Prepare data for LIME
                X_lime = X_transformed
                
                if task_type == "Classification":
                    lime_explainer = LimeTabularExplainer(X_lime, feature_names=feature_names, class_names=[str(c) for c in final_model.classes_], mode="classification" )
                    instance = X_lime[0]

                    lime_explanation = lime_explainer.explain_instance( instance, final_model.predict_proba, num_features=10 )

                    lime_df = pd.DataFrame( lime_explanation.as_list(), columns=["Feature", "Contribution"])

                    st.dataframe(lime_df)
                    lime_available = True


                else:
                    lime_explainer = LimeTabularExplainer( X_lime, feature_names=feature_names, mode="regression")

                # Explain the first prediction
                    instance = X_lime[0]

                    lime_explanation = lime_explainer.explain_instance(instance, final_model.predict, num_features=10 )

                    lime_df = pd.DataFrame( lime_explanation.as_list(), columns=["Feature", "Contribution"])

                    st.dataframe(lime_df)
                    lime_available = True

                st.header("6. Data Quality Assessment")

                total_rows = len(evaluation_df)
                total_columns = len(evaluation_df.columns)
                missing_values = evaluation_df.isnull().sum().sum()
                duplicate_rows = evaluation_df.duplicated().sum()
                st.write("Total Rows:", total_rows)
                st.write("Total Columns:", total_columns)
                st.write("Missing Values:", missing_values)
                st.write("Duplicate Rows:", duplicate_rows)
                
                missing_percentage = (  missing_values / (total_rows * total_columns)) * 100
                duplicate_percentage = (duplicate_rows / total_rows) * 100

                st.write("Missing Value Percentage:", round(missing_percentage, 2), "%")
                st.write("Duplicate Row Percentage:", round(duplicate_percentage, 2), "%")
                data_quality_score = 100

                # Penalty for missing values
                data_quality_score -= missing_percentage

                # Penalty for duplicate rows
                data_quality_score -= duplicate_percentage

                # Keep score between 0 and 100
                data_quality_score = max(0, min(100, data_quality_score))

                st.subheader("Data Quality Score")
                st.write("Score:", round(data_quality_score, 2), "/ 100")



                st.header("7. Stability Assessment")

                split_indices = np.array_split(np.arange(len(X)),5)

                stability_scores = []

                for indices in split_indices:

                    X_part = X.iloc[indices]
                    y_part = y.iloc[indices]

                    part_predictions = model.predict(X_part)

                    if task_type == "Classification":part_score = accuracy_score(y_part,part_predictions)
                    else:
                        part_score = r2_score(y_part,part_predictions)

                    stability_scores.append(part_score)

                st.write("Scores Across 5 Data Segments:",[round(score, 4) for score in stability_scores])
                mean_stability = np.mean(stability_scores)
                std_stability = np.std(stability_scores)

                st.write("Mean Performance:",round(mean_stability, 4))
                st.write("Performance Variation (Std Dev):",round(std_stability, 4))



                st.header("8. Fairness Assessment")

                fairness_attribute = st.selectbox("Select a sensitive attribute (optional)",["Not Selected"] + list(X.columns))

                if fairness_attribute == "Not Selected":
                    st.info("Fairness has not been evaluated.")

                else:
                    st.write("Selected Sensitive Attribute:",fairness_attribute)

                    groups = X[fairness_attribute].dropna().unique()

                    fairness_results = []

                    for group in groups:
                        group_mask = X[fairness_attribute] == group
                        X_group = X[group_mask]
                        y_group = y[group_mask]
                        group_predictions = model.predict(X_group)

                        if task_type == "Classification":
                            group_score = accuracy_score(y_group,group_predictions)

                        else:
                            group_score = mean_absolute_error(y_group,group_predictions)

                        metric_name = ("Accuracy"
                            if task_type == "Classification"
                            else "MAE")

                        fairness_results.append({
                            "Group": group,
                            metric_name: round(group_score, 4),
                            "Samples": len(y_group)})

                    fairness_df = pd.DataFrame(fairness_results)

                    st.subheader("Group Performance")
                    st.dataframe(fairness_df)
                    valid_groups = fairness_df[fairness_df["Samples"] >= 30]

                    if len(valid_groups) >= 2:

                        fairness_gap = (valid_groups[metric_name].max()- valid_groups[metric_name].min())

                        st.write(f"{metric_name} Gap Between Groups:",round(fairness_gap, 4))
                    else:
                        fairness_gap = None

                        st.warning("Not enough group data to calculate a reliable gap.")
                    


                    st.header("9. Trust Score")
                    if task_type == "Classification":
                        performance_score = np.mean([accuracy,precision, recall, f1, roc_auc]) * 100
                    else:
                        performance_score = max( 0, min(100, r2 * 100))

                    st.subheader("Performance Score")

                    st.write("Score:",round(performance_score, 2),"/ 100")
                    
                    comparison_performance_score = None

                    if comparison_model is not None:

                        if task_type == "Classification":
                            comparison_accuracy = accuracy_score( y, comparison_predictions)
                            comparison_precision = precision_score( y, comparison_predictions)
                            comparison_recall = recall_score( y, comparison_predictions)
                            comparison_f1 = f1_score( y, comparison_predictions)

                            comparison_probabilities = comparison_model.predict_proba(X)[:, 1]

                            comparison_roc_auc = roc_auc_score( y, comparison_probabilities)

                            comparison_performance_score = np.mean([ comparison_accuracy, comparison_precision, comparison_recall, comparison_f1, comparison_roc_auc]) * 100

                        else:
                            comparison_performance_score = max( 0, min(100, comparison_r2 * 100) )

                        st.write( "Comparison Model Score:", round(comparison_performance_score, 2), "/ 100" )

                    st.subheader("Data Quality Score")
                    st.write("Score:",round(data_quality_score, 2),"/ 100")

                    st.subheader("Stability Score")

                    if mean_stability != 0:
                        relative_variation = (std_stability / abs(mean_stability))

                        stability_score = (1 - relative_variation) * 100

                        stability_score = max(0,min(100, stability_score))
                    else:
                        stability_score = 0
                    st.write("Score:",round(stability_score, 2),"/ 100")
                    
                    comparison_stability_score = None

                    if comparison_model is not None:

                        comparison_stability_scores = []

                        for indices in split_indices:

                            X_part = X.iloc[indices]
                            y_part = y.iloc[indices]

                            comparison_part_predictions = comparison_model.predict(X_part)

                            if task_type == "Classification":
                                comparison_part_score = accuracy_score( y_part, comparison_part_predictions)
                            else:
                                comparison_part_score = r2_score( y_part, comparison_part_predictions)

                            comparison_stability_scores.append(comparison_part_score)

                        comparison_mean_stability = np.mean(comparison_stability_scores)

                        comparison_std_stability = np.std( comparison_stability_scores)

                        if comparison_mean_stability != 0:

                            comparison_relative_variation = (comparison_std_stability / abs(comparison_mean_stability) )

                            comparison_stability_score = ( 1 - comparison_relative_variation ) * 100

                            comparison_stability_score = max( 0, min(100, comparison_stability_score))

                        else:
                            comparison_stability_score = 0

                        st.write("Comparison Model Score:",round(comparison_stability_score, 2),"/ 100")


                    st.subheader("Explainability Coverage Score")
                    explainability_score = 0
                    if shap_available:
                        explainability_score += 50
                    if lime_available:
                        explainability_score += 50

                    st.write("Score:", explainability_score, "/ 100")
                    st.caption("Coverage: 50 points for SHAP + 50 points for LIME")

                    comparison_shap_available = False
                    comparison_lime_available = False

                    if comparison_model is not None:

                        try:
                            comparison_preprocessor = comparison_model.named_steps["preprocessor"]
                            comparison_final_model = comparison_model.named_steps["model"]

                            comparison_X_transformed = comparison_preprocessor.transform(X)

                            if hasattr(comparison_X_transformed, "toarray"):
                                comparison_X_transformed = comparison_X_transformed.toarray()

                            comparison_X_shap = comparison_X_transformed[:100]

        # SHAP check
                            if type(comparison_final_model).__name__ in [
                                "RandomForestRegressor", "RandomForestClassifier", "DecisionTreeRegressor", "DecisionTreeClassifier" ]:
                                comparison_explainer = shap.TreeExplainer( comparison_final_model)
                                comparison_explainer.shap_values( comparison_X_shap )
                                comparison_shap_available = True

                            elif type(comparison_final_model).__name__ in ["LinearRegression", "LogisticRegression"]:
                                comparison_explainer = shap.LinearExplainer( comparison_final_model, comparison_X_transformed)
                                comparison_explainer.shap_values( comparison_X_shap )
                                comparison_shap_available = True

        # LIME check
                            comparison_lime_explainer = LimeTabularExplainer( comparison_X_transformed,
                                mode=( "classification"
                                    if task_type == "Classification"
                                    else "regression" ) )

                            if task_type == "Classification":
                                comparison_lime_explainer.explain_instance( comparison_X_transformed[0], comparison_final_model.predict_proba, num_features=10 )
                            else:
                                comparison_lime_explainer.explain_instance( comparison_X_transformed[0], comparison_final_model.predict, num_features=10 )

                            comparison_lime_available = True

                        except Exception as e:
                            st.warning( "Comparison model explainability could not be fully evaluated." )

                    comparison_explainability_score = 0

                    if comparison_shap_available:
                        comparison_explainability_score += 50

                    if comparison_lime_available:
                        comparison_explainability_score += 50

                    if comparison_model is not None:
                        st.write( "Comparison Model Explainability Coverage:", comparison_explainability_score, "/ 100" )

                    
                    st.subheader("Fairness Score")

                    if fairness_attribute == "Not Selected":
                        fairness_score = None
                        st.write("Score: Not Evaluated")
                    elif fairness_gap is not None:
                        if task_type == "Classification":
                            fairness_score = (1 - fairness_gap) * 100
                        else:
                            relative_fairness_gap = (fairness_gap / mae)
                            fairness_score = (1 - relative_fairness_gap) * 100
                        fairness_score = max(0,min(100, fairness_score))
                        st.write("Score:",round(fairness_score, 2),"/ 100")
                    else:
                        fairness_score = None
                        st.write("Score: Insufficient Data")
 
# Comparison Model Fairness Score
                    comparison_fairness_score = None

                    if comparison_model is not None and fairness_attribute != "Not Selected":

                        comparison_fairness_results = []

                        for group in groups:

                            group_mask = X[fairness_attribute] == group
                            X_group = X[group_mask]
                            y_group = y[group_mask]

                            comparison_group_predictions = comparison_model.predict(X_group)

                            if task_type == "Classification":
                                comparison_group_score = accuracy_score( y_group, comparison_group_predictions )
                            else:
                                comparison_group_score = mean_absolute_error(y_group, comparison_group_predictions )

                            comparison_fairness_results.append({ "Group": group, "Score": comparison_group_score, "Samples": len(y_group)})

                        comparison_fairness_df = pd.DataFrame(comparison_fairness_results)

                        comparison_valid_groups = comparison_fairness_df[ comparison_fairness_df["Samples"] >= 30 ]

                        if len(comparison_valid_groups) >= 2:

                            comparison_fairness_gap = ( comparison_valid_groups["Score"].max() - comparison_valid_groups["Score"].min() )

                            if task_type == "Classification":

                                comparison_fairness_score = ( 1 - comparison_fairness_gap ) * 100

                            else:

                                comparison_relative_fairness_gap = ( comparison_fairness_gap / comparison_mae )

                                comparison_fairness_score = ( 1 - comparison_relative_fairness_gap ) * 100

                            comparison_fairness_score = max( 0, min(100, comparison_fairness_score))

                            st.write( "Comparison Model Score:", round(comparison_fairness_score, 2), "/ 100")

                    st.subheader("Overall Trust Score")
                    if fairness_score is not None:
                        trust_score = (performance_score * 0.35 + data_quality_score * 0.20 + stability_score * 0.20 + explainability_score * 0.15 + fairness_score * 0.10)

                    else:
                        trust_score = ( performance_score * 0.40 + data_quality_score * 0.25 + stability_score * 0.20 + explainability_score * 0.15 )
                    st.write( "Trust Score:", round(trust_score, 2), "/ 100")

                    if comparison_model is not None:

                        if comparison_fairness_score is not None:
                            comparison_trust_score = (comparison_performance_score * 0.35
            + data_quality_score * 0.20
            + comparison_stability_score * 0.20
            + comparison_explainability_score * 0.15
            + comparison_fairness_score * 0.10)
                        else:
                            comparison_trust_score = (comparison_performance_score * 0.40
            + data_quality_score * 0.25
            + comparison_stability_score * 0.20
            + comparison_explainability_score * 0.15)

                        st.write("Comparison Model Trust Score:", round(comparison_trust_score, 2), "/ 100")
                        if comparison_model is not None:

                            st.subheader("Multi-Model Trust Comparison")

                            primary_name = type(model.named_steps["model"]).__name__
                            comparison_name = type(comparison_model.named_steps["model"]).__name__

                            trust_comparison_df = pd.DataFrame({
                                "Component": [ "Performance", "Data Quality", "Stability", "Explainability Coverage", "Fairness", "Overall Trust Score"],
                                primary_name: [ round(performance_score, 2), round(data_quality_score, 2), round(stability_score, 2), round(explainability_score, 2),
 round(fairness_score, 2) if fairness_score is not None else "Not Evaluated", round(trust_score, 2)],
                                comparison_name: [ round(comparison_performance_score, 2), round(data_quality_score, 2), round(comparison_stability_score, 2), round(comparison_explainability_score, 2),
 round(comparison_fairness_score, 2) if comparison_fairness_score is not None else "Not Evaluated", round(comparison_trust_score, 2) ] })

                            st.dataframe(trust_comparison_df, hide_index=True)

                            performance_winner = ( primary_name
            if performance_score > comparison_performance_score
            else comparison_name)

                            trust_winner = ( primary_name
            if trust_score > comparison_trust_score
            else comparison_name)

                            if performance_winner != trust_winner:
                                        st.info(
                "Performance ranking and Trust Score ranking differ. "
                "The model with higher predictive performance is not the same "
                "as the model with the higher overall Trust Score.")
                            else:
                                st.info(
                "Performance ranking and Trust Score ranking are consistent "
                "for these two models.")
                
                    st.header("10. Recommendations")

                    recommendations = []

                    if performance_score < 85:
                        recommendations.append("Model performance can be improved. Consider feature engineering, "
                            "hyperparameter tuning, or testing alternative models.")

                    if data_quality_score < 90:
                        recommendations.append(
                            "Improve dataset quality by handling missing values, duplicates, "
                            "and inconsistent records.")

                    if stability_score < 90:
                        recommendations.append(
                            "Model performance varies across data segments. Investigate data "
                            "distribution differences and model robustness.")

                    if explainability_score < 100:
                        recommendations.append("Improve model explainability by providing additional explanation methods.")

                    if fairness_score is not None and fairness_score < 80:
                        recommendations.append(
                            "Performance differs across the selected groups. Review group-level "
                            "errors, sample sizes, data representation, and relevant features.")

                    if fairness_score is None:
                        recommendations.append(
                            "Fairness was not evaluated. Select an appropriate group attribute "
                            "if group-level assessment is relevant.")

                    if len(recommendations) == 0:
                        st.success( "No major issues were identified by the current TrustAI checks.")

                    else:
                        for recommendation in recommendations:
                            st.warning(recommendation)

        except Exception as e:
            st.error("Dataset is not compatible with the uploaded model.")
            st.warning("Please upload the dataset that matches this trained model.")