import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load example data and train a model (replace with your own model if available)
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
X = df
y = data.target
model = RandomForestClassifier()
model.fit(X, y)

st.title("ML Model Deployment with Streamlit")
st.write("Input data to get predictions and visualize model outputs.")

# User input
with st.form("input_form"):
    sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1)
    sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5)
    petal_length = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4)
    petal_width = st.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=0.2)
    submitted = st.form_submit_button("Predict")

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if submitted:
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    st.success(f"Predicted class: {data.target_names[prediction]}")
    st.write("Prediction probabilities:")
    st.bar_chart(proba)

    # Feature importance visualization
    st.write("Feature Importances:")
    fig, ax = plt.subplots()
    sns.barplot(x=model.feature_importances_, y=df.columns, ax=ax)
    st.pyplot(fig)

st.write("---")
st.write("## Data Preview")
st.dataframe(df.head())
