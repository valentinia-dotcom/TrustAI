import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="TrustAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
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
        "Configure the dataset and trained model used for the TrustAI assessment."
    )

    st.divider()

    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------

    st.subheader("1. Dataset")

    data_file = st.file_uploader(
        "Upload dataset",
        type=["csv"],
        key="main_dataset"
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
                    use_container_width=True
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
                df.columns
            )

            st.session_state.target = target

        with col2:

            task_type = st.selectbox(
                "Machine learning task",
                ["Classification", "Regression"]
            )

            st.session_state.task_type = task_type

    # -----------------------------------------------------
    # HELD-OUT DATA
    # -----------------------------------------------------

        st.subheader("3. Evaluation Dataset")

        test_file = st.file_uploader(
            "Upload held-out evaluation CSV (recommended)",
            type=["csv"],
            key="evaluation_dataset_uploader"
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

            st.session_state.evaluation_dataset = df

            st.warning(
                "No held-out dataset uploaded. TrustAI will use the "
                "uploaded dataset for evaluation. These results should "
                "not be described as held-out performance."
            )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    st.subheader("4. Primary Model")

    model_file = st.file_uploader(
        "Upload trained model",
        type=["pkl", "joblib"],
        key="primary_model"
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
        key="comparison_model"
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

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Dataset",
        "Ready" if dataset_ready else "Missing"
    )

    c2.metric(
        "Model",
        "Ready" if model_ready else "Missing"
    )

    c3.metric(
        "Configuration",
        "Ready" if target_ready else "Missing"
    )

    if dataset_ready and model_ready and target_ready:

        st.success(
            "Setup complete. Continue to Model Audit."
        )


# =========================================================
# PLACEHOLDER PAGES
# =========================================================

elif page == "📊 Model Audit":

    st.title("📊 Model Audit")

    st.info(
        "Performance, data quality, and stability analysis "
        "will appear here."
    )


elif page == "🔍 Explainability":

    st.title("🔍 Explainability")

    st.info(
        "Permutation importance, SHAP, and LIME explanations "
        "will appear here."
    )


elif page == "⚖️ Fairness":

    st.title("⚖️ Fairness")

    st.info(
        "Group-level fairness diagnostics will appear here."
    )


elif page == "🔄 Model Comparison":

    st.title("🔄 Model Comparison")

    st.info(
        "Side-by-side model comparison will appear here."
    )


elif page == "📄 Trust Report":

    st.title("📄 Trust Report")

    st.info(
        "The final Trust Score, risk summary, and recommendations "
        "will appear here."
    )
