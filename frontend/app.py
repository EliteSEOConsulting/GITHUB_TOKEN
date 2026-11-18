import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860").rstrip("/")

st.title("SuperKart System")

single_tab, batch_tab = st.tabs(["Single prediction", "Batch predictions"])

with single_tab:
    with st.form("single_prediction"):
        left, right = st.columns(2)

        with left:
            product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
            sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
            allocated_area = st.number_input(
                "Product Allocated Area", min_value=0.0, value=0.027, format="%.3f"
            )
            product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08)
            product_id = st.selectbox("Product ID Prefix", ["DR", "FD", "NC"])

        with right:
            store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
            city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
            store_type = st.selectbox(
                "Store Type",
                ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"],
            )
            store_age = st.number_input("Store Age (Years)", min_value=0, value=10)
            product_category = st.selectbox(
                "Product Type Category", ["Perishables", "Non Perishables"]
            )

        predict = st.form_submit_button("Predict sales")

    if predict:
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar_content,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": product_mrp,
            "Product_Id_char": product_id,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Store_Age_Years": store_age,
            "Product_Type_Category": product_category,
        }

        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            response.raise_for_status()
            sales = response.json()["Sales"]
            st.metric("Predicted sales", f"{sales:,.2f}")
        except (requests.RequestException, KeyError, ValueError) as error:
            st.error(f"Prediction failed. {error}")

with batch_tab:
    csv_file = st.file_uploader("Upload a CSV with the model input columns", type=["csv"])

    if st.button("Predict batch", disabled=csv_file is None):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": (csv_file.name, csv_file.getvalue(), "text/csv")},
                timeout=60,
            )
            response.raise_for_status()
            predictions = response.json()
            results = pd.DataFrame(
                [
                    {"CSV row": int(row) + 1, "Predicted sales": sales}
                    for row, sales in predictions.items()
                ]
            ).sort_values("CSV row")
            st.dataframe(results, hide_index=True)
        except (requests.RequestException, KeyError, ValueError) as error:
            st.error(f"Batch prediction failed. {error}")
