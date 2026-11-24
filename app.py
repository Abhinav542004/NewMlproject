
import streamlit as st
import pandas as pd

from inference.predict_freight import predict_freight_cost, load_model as load_freight_model
from inference.predict_invoice_flag import predict_invoice_flag, load_model as load_invoice_model, load_scaler


# --------------------------------------------------
# CACHED MODEL LOADING  (runs once at startup)
# --------------------------------------------------

@st.cache_resource
def get_freight_model():
    """Load freight model once and cache it for the session."""
    return load_freight_model()


@st.cache_resource
def get_invoice_model():
    """Load invoice risk model once and cache it for the session."""
    return load_invoice_model()


@st.cache_resource
def get_invoice_scaler():
    """Load invoice scaler once and cache it for the session."""
    return load_scaler()


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# HEADER SECTION
# --------------------------------------------------

st.title("📊 Vendor Invoice Intelligence Portal")

st.subheader(
    "AI-Driven Freight Cost Prediction & Invoice Risk Flagging"
)

st.markdown("""
This internal analytics portal leverages machine learning to:

- **Forecast freight costs**
- **Detect potentially risky or abnormal vendor invoices**
- **Reduce financial leakage and manual workload**
- **Support faster finance operations**
""")

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("""
### Business Impact

- Improved cost forecasting
- Reduced invoice anomalies
- Faster finance operations
- Better financial monitoring
""")


# ==================================================
# MODULE 1: FREIGHT COST PREDICTION
# ==================================================

if selected_model == "Freight Cost Prediction":

    st.header("🚚 Freight Cost Prediction")

    st.markdown("""
    **Objective:**

    Predict freight costs for vendor invoices using
    invoice dollar amounts to support budgeting,
    forecasting, and vendor negotiations.
    """)

    st.info(
        "The current trained Linear Regression model "
        "uses Invoice Dollars as its input feature."
    )

    with st.form("freight_form"):

        st.subheader("Enter Invoice Details")

        dollars = st.number_input(
            "Invoice Dollars ($)",
            min_value=1.0,
            value=18500.0,
            step=100.0
        )

        submit_freight = st.form_submit_button(
            "🔮 Predict Freight Cost",
            use_container_width=True
        )

    if submit_freight:

        input_data = {
            "Dollars": [dollars]
        }

        try:
            prediction = predict_freight_cost(input_data, model=get_freight_model())

            predicted_freight = prediction[
                "Predicted_Freight"
            ].iloc[0]

            st.success("Freight prediction completed!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Invoice Amount",
                    f"${dollars:,.2f}"
                )

            with col2:
                st.metric(
                    "Estimated Freight Cost",
                    f"${predicted_freight:,.2f}"
                )

            st.subheader("Prediction Details")

            st.dataframe(
                prediction,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ==================================================
# MODULE 2: INVOICE RISK FLAGGING
# ==================================================

else:

    st.header("🛡️ Invoice Manual Approval Prediction")

    st.markdown("""
    **Objective:**

    Predict whether a vendor invoice should be flagged
    for manual review based on invoice amounts,
    freight costs, and purchase information.
    """)

    with st.form("invoice_flag_form"):

        st.subheader("Enter Invoice Details")

        col1, col2, col3 = st.columns(3)

        with col1:

            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )

            invoice_dollars = st.number_input(
                "Invoice Dollars ($)",
                min_value=1.0,
                value=352.95
            )

        with col2:

            freight = st.number_input(
                "Freight Cost ($)",
                min_value=0.0,
                value=1.73
            )

            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )

        with col3:

            total_item_dollars = st.number_input(
                "Total Item Dollars ($)",
                min_value=1.0,
                value=2476.0
            )

        submit_flag = st.form_submit_button(
            "🔍 Evaluate Invoice Risk",
            use_container_width=True
        )

    if submit_flag:

        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        try:

            flag_prediction = predict_invoice_flag(
                input_data,
                model=get_invoice_model(),
                scaler=get_invoice_scaler()
            )

            prediction = flag_prediction[
                "Predicted_Flag"
            ].iloc[0]

            probability = flag_prediction[
                "Risk_Probability"
            ].iloc[0]

            st.divider()

            st.subheader("Invoice Risk Assessment")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Risk Probability",
                    f"{probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Prediction",
                    "Flagged" if prediction == 1
                    else "Not Flagged"
                )

            if prediction == 1:

                st.error(
                    "⚠️ Invoice requires MANUAL APPROVAL"
                )

                st.warning(
                    "This invoice has been identified as "
                    "potentially risky by the classification model."
                )

            else:

                st.success(
                    "✅ Invoice NOT FLAGGED by the model"
                )

                st.info(
                    "Continue with your organization's "
                    "normal invoice approval controls."
                )

            st.subheader("Invoice Prediction Details")

            st.dataframe(
                flag_prediction,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(f"Invoice prediction failed: {e}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Vendor Invoice Intelligence Portal | "
    "Powered by Machine Learning | "
    "Freight Prediction & Invoice Risk Classification"
)
