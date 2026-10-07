import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="TrustAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# MODEL LOADER
# =========================================================

@st.cache_resource
def load_model(model_file):
    return joblib.load(model_file)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "dataset": None,
    "evaluation_dataset": None,
    "model": None,
    "comparison_model": None,
    "target": None,
    "task_type": None,
    "performance_results": None,
    "performance_score": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🛡️ TrustAI")

    st.caption("Machine Learning Trustworthiness Assessment")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "⚙️ Setup & Upload",
            "📊 Model Audit",
            "🔍 Explainability",
            "⚖️ Fairness",
            "🔄 Model Comparison",
            "📄 Trust Report",
        ],
    )

    st.divider()

    st.caption(
        "TrustAI evaluates multiple dimensions of model "
        "trustworthiness. Results should support — not replace — "
        "human review."
    )


# =========================================================
# OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.title("🛡️ TrustAI")

    st.subheader(
        "Explainable Machine Learning Trustworthiness Assessment"
    )

    st.write(
        """
        TrustAI is a model-audit platform for evaluating trained
        machine learning models beyond predictive performance.

        It combines several diagnostic dimensions to provide a
        structured view of model behaviour and potential risks.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            """
            ### 📊 Model Audit

            Evaluate predictive performance, data quality,
            and model stability.
            """
        )

    with col2:
        st.info(
            """
            ### 🔍 Explainability

            Inspect feature importance and model behaviour
            using interpretable ML techniques.
            """
        )

    with col3:
        st.info(
            """
            ### ⚖️ Fairness

            Compare model behaviour across selected
            groups when appropriate.
            """
        )

    col4, col5 = st.columns(2)

    with col4:
        st.info(
            """
            ### 🔄 Model Comparison

            Compare two trained models across multiple
            trust-related dimensions.
            """
        )

    with col5:
        st.info(
            """
            ### 📄 Trust Report

            Summarize findings, risk indicators,
            and recommended actions.
            """
        )

    st.divider()

    st.warning(
        """
        **Important:** The Trust Score is a configurable model-audit
        indicator derived from selected metrics and weights. It should
        not be interpreted as a universal or certified measure of AI
        trustworthiness.
        """
    )


# =========================================================
# SETUP & UPLOAD
# =========================================================

elif page == "⚙️ Setup & Upload":

    st.title("⚙️ Setup & Upload")

    st.write(
        "Configure the dataset and trained model used for "
        "the TrustAI assessment."
    )

    st.divider()

    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------

    st.subheader("1. Dataset")

    data_file = st.file_uploader(
        "Upload dataset",
        type=["csv"],
        key="main_dataset_uploader",
    )

    if data_file is not None:

        try:
            df = pd.read_csv(data_file)

            st.session_state.dataset = df

            st.success(
                f"Dataset loaded — {df.shape[0]} rows × "
                f"{df.shape[1]} columns"
            )

            with st.expander("Preview dataset"):
                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Could not read dataset: {e}")

    # -----------------------------------------------------
    # CONFIGURATION
    # -----------------------------------------------------

    if st.session_state.dataset is not None:

        df = st.session_state.dataset

        st.subheader("2. Assessment Configuration")

        col1, col2 = st.columns(2)

        with col1:

            target = st.selectbox(
                "Target column",
                df.columns,
                key="target_selector",
            )

            st.session_state.target = target

        with col2:

            task_type = st.selectbox(
                "Machine learning task",
                ["Classification", "Regression"],
                key="task_type_selector",
            )

            st.session_state.task_type = task_type

        # -----------------------------------------------------
        # HELD-OUT DATA
        # -----------------------------------------------------

        st.subheader("3. Evaluation Dataset")

        test_file = st.file_uploader(
            "Upload held-out evaluation CSV (recommended)",
            type=["csv"],
            key="evaluation_dataset_uploader",
        )

        if test_file is not None:

            try:
                test_df = pd.read_csv(test_file)

                if target not in test_df.columns:

                    st.error(
                        f"Held-out dataset does not contain "
                        f"target column '{target}'."
                    )

                else:

                    st.session_state.evaluation_dataset = test_df

                    st.success(
                        f"Held-out dataset loaded — "
                        f"{len(test_df)} rows"
                    )

            except Exception as e:

                st.error(
                    f"Could not read held-out dataset: {e}"
                )

        else:

            st.session_state.evaluation_dataset = None

            st.warning(
                "No held-out dataset uploaded. TrustAI will use the "
                "main uploaded dataset for evaluation. These results "
                "should not be described as held-out performance."
            )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    st.subheader("4. Primary Model")

    model_file = st.file_uploader(
        "Upload trained model",
        type=["pkl", "joblib"],
        key="primary_model_uploader",
    )

    if model_file is not None:

        try:

            model = load_model(model_file)

            st.session_state.model = model

            st.success(
                f"Model loaded: {type(model).__name__}"
            )

        except Exception as e:

            st.error(
                f"Could not load model: {e}"
            )

    # -----------------------------------------------------
    # COMPARISON MODEL
    # -----------------------------------------------------

    st.subheader("5. Comparison Model")

    comparison_file = st.file_uploader(
        "Upload another model (optional)",
        type=["pkl", "joblib"],
        key="comparison_model_uploader",
    )

    if comparison_file is not None:

        try:

            comparison_model = load_model(comparison_file)

            st.session_state.comparison_model = comparison_model

            st.success(
                f"Comparison model loaded: "
                f"{type(comparison_model).__name__}"
            )

        except Exception as e:

            st.error(
                f"Could not load comparison model: {e}"
            )

    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    st.divider()

    st.subheader("Assessment Status")

    dataset_ready = st.session_state.dataset is not None
    model_ready = st.session_state.model is not None
    target_ready = st.session_state.target is not None
    task_ready = st.session_state.task_type is not None

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Dataset",
        "Ready" if dataset_ready else "Missing",
    )

    c2.metric(
        "Model",
        "Ready" if model_ready else "Missing",
    )

    c3.metric(
        "Configuration",
        "Ready"
        if target_ready and task_ready
        else "Missing",
    )

    if (
        dataset_ready
        and model_ready
        and target_ready
        and task_ready
    ):

        st.success(
            "Setup complete. Continue to Model Audit."
        )


# =========================================================
# MODEL AUDIT
# =========================================================

elif page == "📊 Model Audit":

    st.title("📊 Model Audit")

    st.write(
        "Evaluate the model's predictive performance using "
        "the configured evaluation dataset."
    )

    # -----------------------------------------------------
    # CHECK SETUP
    # -----------------------------------------------------

    if (
        st.session_state.dataset is None
        or st.session_state.model is None
        or st.session_state.target is None
        or st.session_state.task_type is None
    ):

        st.warning(
            "Setup is incomplete. Go to **Setup & Upload** and "
            "provide the dataset, target column, task type, "
            "and primary model."
        )

        st.stop()

    model = st.session_state.model
    target = st.session_state.target
    task_type = st.session_state.task_type

    # -----------------------------------------------------
    # EVALUATION DATASET
    # -----------------------------------------------------

    if st.session_state.evaluation_dataset is not None:

        eval_df = (
            st.session_state.evaluation_dataset.copy()
        )

        dataset_source = "Held-out evaluation dataset"
        using_held_out = True

    else:

        eval_df = st.session_state.dataset.copy()

        dataset_source = "Main uploaded dataset"
        using_held_out = False

    st.subheader("1. Evaluation Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dataset",
        dataset_source,
    )

    col2.metric(
        "Rows",
        len(eval_df),
    )

    col3.metric(
        "Columns",
        len(eval_df.columns),
    )

    if using_held_out:

        st.success(
            "Performance is being evaluated on the "
            "held-out dataset."
        )

    else:

        st.warning(
            "No held-out evaluation dataset was provided. "
            "Performance is being measured on the main dataset, "
            "so the results may be optimistic."
        )

    # -----------------------------------------------------
    # TARGET VALIDATION
    # -----------------------------------------------------

    if target not in eval_df.columns:

        st.error(
            f"The selected target column '{target}' is not "
            "present in the evaluation dataset."
        )

        st.stop()

    X_eval = eval_df.drop(columns=[target])
    y_eval = eval_df[target]

    # -----------------------------------------------------
    # FEATURE COMPATIBILITY
    # -----------------------------------------------------

    expected_features = None

    if hasattr(model, "feature_names_in_"):

        expected_features = list(
            model.feature_names_in_
        )

    if expected_features is not None:

        missing_features = [
            feature
            for feature in expected_features
            if feature not in X_eval.columns
        ]

        extra_features = [
            feature
            for feature in X_eval.columns
            if feature not in expected_features
        ]

        if missing_features:

            st.error(
                "The evaluation dataset is missing features "
                "required by the model:"
            )

            st.write(missing_features)

            st.stop()

        X_model = X_eval[expected_features]

        if extra_features:

            st.info(
                "The evaluation dataset contains additional "
                "columns that are not required by the model. "
                "They will be ignored."
            )

    else:

        X_model = X_eval

        st.info(
            "The model does not expose `feature_names_in_`. "
            "TrustAI therefore cannot automatically verify "
            "the original feature names."
        )

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    st.divider()

    st.subheader("2. Predictive Performance")

    try:

        y_pred = model.predict(X_model)

    except Exception as e:

        st.error(
            "The model could not generate predictions."
        )

        st.exception(e)

        st.stop()

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if task_type == "Classification":

        try:

            accuracy = accuracy_score(
                y_eval,
                y_pred,
            )

            precision = precision_score(
                y_eval,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            recall = recall_score(
                y_eval,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            f1 = f1_score(
                y_eval,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            roc_auc = None

            if hasattr(model, "predict_proba"):

                try:

                    probabilities = (
                        model.predict_proba(X_model)
                    )

                    classes = getattr(
                        model,
                        "classes_",
                        np.unique(y_eval),
                    )

                    if len(classes) == 2:

                        roc_auc = roc_auc_score(
                            y_eval,
                            probabilities[:, 1],
                        )

                    elif len(classes) > 2:

                        roc_auc = roc_auc_score(
                            y_eval,
                            probabilities,
                            multi_class="ovr",
                            average="weighted",
                            labels=classes,
                        )

                except Exception:

                    roc_auc = None

            metric_cols = st.columns(5)

            metric_cols[0].metric(
                "Accuracy",
                f"{accuracy:.3f}",
            )

            metric_cols[1].metric(
                "Precision",
                f"{precision:.3f}",
            )

            metric_cols[2].metric(
                "Recall",
                f"{recall:.3f}",
            )

            metric_cols[3].metric(
                "F1 Score",
                f"{f1:.3f}",
            )

            metric_cols[4].metric(
                "ROC-AUC",
                f"{roc_auc:.3f}"
                if roc_auc is not None
                else "N/A",
            )

            scoring_metrics = [
                accuracy,
                precision,
                recall,
                f1,
            ]

            if roc_auc is not None:
                scoring_metrics.append(roc_auc)

            performance_score = (
                sum(scoring_metrics)
                / len(scoring_metrics)
            ) * 100

            performance_score = max(
                0,
                min(100, performance_score),
            )

            # ---------------------------------------------
            # CONFUSION MATRIX
            # ---------------------------------------------

            st.subheader("Confusion Matrix")

            labels = list(
                np.unique(
                    np.concatenate(
                        [
                            np.asarray(y_eval),
                            np.asarray(y_pred),
                        ]
                    )
                )
            )

            cm = confusion_matrix(
                y_eval,
                y_pred,
                labels=labels,
            )

            cm_df = pd.DataFrame(
                cm,
                index=[
                    f"Actual {label}"
                    for label in labels
                ],
                columns=[
                    f"Predicted {label}"
                    for label in labels
                ],
            )

            st.dataframe(
                cm_df,
                use_container_width=True,
            )

            # ---------------------------------------------
            # CLASSIFICATION REPORT
            # ---------------------------------------------

            with st.expander(
                "View Classification Report"
            ):

                report = classification_report(
                    y_eval,
                    y_pred,
                    output_dict=True,
                    zero_division=0,
                )

                report_df = (
                    pd.DataFrame(report).transpose()
                )

                st.dataframe(
                    report_df,
                    use_container_width=True,
                )

            performance_results = {
                "task_type": "Classification",
                "dataset_source": dataset_source,
                "held_out": using_held_out,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": (
                    float(roc_auc)
                    if roc_auc is not None
                    else None
                ),
                "performance_score": float(
                    performance_score
                ),
            }

        except Exception as e:

            st.error(
                "An error occurred while calculating "
                "classification metrics."
            )

            st.exception(e)

            st.stop()

    # =====================================================
    # REGRESSION
    # =====================================================

    elif task_type == "Regression":

        try:

            r2 = r2_score(
                y_eval,
                y_pred,
            )

            mae = mean_absolute_error(
                y_eval,
                y_pred,
            )

            mse = mean_squared_error(
                y_eval,
                y_pred,
            )

            rmse = np.sqrt(mse)

            y_range = (
                float(y_eval.max())
                - float(y_eval.min())
            )

            if y_range > 0:

                normalized_rmse = (
                    rmse / y_range
                )

            else:

                normalized_rmse = 0.0

            # ---------------------------------------------
            # PERFORMANCE INDICATOR
            #
            # 50% R² component
            # 50% normalized RMSE component
            # ---------------------------------------------

            r2_component = max(
                0,
                min(1, r2),
            )

            error_component = max(
                0,
                min(
                    1,
                    1 - normalized_rmse,
                ),
            )

            performance_score = (
                (0.50 * r2_component)
                + (0.50 * error_component)
            ) * 100

            performance_score = max(
                0,
                min(100, performance_score),
            )

            metric_cols = st.columns(4)

            metric_cols[0].metric(
                "R²",
                f"{r2:.3f}",
            )

            metric_cols[1].metric(
                "MAE",
                f"{mae:,.3f}",
            )

            metric_cols[2].metric(
                "RMSE",
                f"{rmse:,.3f}",
            )

            metric_cols[3].metric(
                "Normalized RMSE",
                f"{normalized_rmse:.3f}",
            )

            performance_results = {
                "task_type": "Regression",
                "dataset_source": dataset_source,
                "held_out": using_held_out,
                "r2": float(r2),
                "mae": float(mae),
                "rmse": float(rmse),
                "normalized_rmse": float(
                    normalized_rmse
                ),
                "performance_score": float(
                    performance_score
                ),
            }

        except Exception as e:

            st.error(
                "An error occurred while calculating "
                "regression metrics."
            )

            st.exception(e)

            st.stop()

    else:

        st.error(
            "Unknown task type. Select Classification or "
            "Regression in Setup & Upload."
        )

        st.stop()

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    st.session_state.performance_results = (
        performance_results
    )

    st.session_state.performance_score = (
        performance_score
    )

    # -----------------------------------------------------
    # PERFORMANCE INDICATOR
    # -----------------------------------------------------

    st.divider()

    st.subheader("3. Performance Indicator")

    score_col, risk_col = st.columns(2)

    score_col.metric(
        "Performance Score",
        f"{performance_score:.1f} / 100",
    )

    if performance_score >= 80:

        performance_level = "Strong"
        risk_level = "Low"

    elif performance_score >= 60:

        performance_level = "Moderate"
        risk_level = "Medium"

    else:

        performance_level = "Weak"
        risk_level = "High"

    risk_col.metric(
        "Performance Risk",
        risk_level,
    )

    if performance_score >= 80:

        st.success(
            "Strong predictive performance was observed "
            "on the selected evaluation dataset."
        )

    elif performance_score >= 60:

        st.warning(
            "Moderate predictive performance was observed. "
            "Review individual metrics before relying on "
            "the model."
        )

    else:

        st.error(
            "Weak predictive performance was observed. "
            "The model may require further validation "
            "or improvement."
        )

    # -----------------------------------------------------
    # AUDIT CONTEXT
    # -----------------------------------------------------

    st.subheader("4. Audit Context")

    context_df = pd.DataFrame(
        {
            "Item": [
                "Task Type",
                "Evaluation Source",
                "Held-out Evaluation",
                "Evaluation Rows",
                "Performance Level",
            ],
            "Value": [
                task_type,
                dataset_source,
                (
                    "Yes"
                    if using_held_out
                    else "No"
                ),
                len(eval_df),
                performance_level,
            ],
        }
    )

    st.dataframe(
        context_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The Performance Score is a configurable audit "
        "indicator derived from selected predictive metrics. "
        "It is intended to support model review and should "
        "not be interpreted as a universal or certified "
        "measure of model quality."
    )


# =========================================================
# EXPLAINABILITY
# =========================================================

elif page == "🔍 Explainability":

    st.title("🔍 Explainability")

    st.info(
        "Permutation importance, SHAP, and LIME explanations "
        "will appear here."
    )


# =========================================================
# FAIRNESS
# =========================================================

elif page == "⚖️ Fairness":

    st.title("⚖️ Fairness")

    st.info(
        "Group-level fairness diagnostics will appear here."
    )


# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "🔄 Model Comparison":

    st.title("🔄 Model Comparison")

    st.info(
        "Side-by-side model comparison will appear here."
    )


# =========================================================
# TRUST REPORT
# =========================================================

elif page == "📄 Trust Report":

    st.title("📄 Trust Report")

    if st.session_state.performance_results is None:

        st.info(
            "Complete the Model Audit to begin building "
            "the Trust Report."
        )

    else:

        st.subheader("Current Audit Results")

        c1, c2 = st.columns(2)

        c1.metric(
            "Performance Score",
            f"{st.session_state.performance_score:.1f} / 100",
        )

        held_out = (
            st.session_state.performance_results.get(
                "held_out",
                False,
            )
        )

        c2.metric(
            "Evaluation",
            (
                "Held-out"
                if held_out
                else "Main Dataset"
            ),
        )

        st.info(
            "The complete Trust Report will combine "
            "performance, data quality, stability, "
            "explainability, and fairness indicators "
            "as those modules are completed."
        )
