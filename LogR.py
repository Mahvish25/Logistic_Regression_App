from imblearn.over_sampling import SMOTE
import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

st.title("Logistic Regression on Purchase Prediction")

upload_file = st.file_uploader("Upload your csv file", type=["csv"])
if upload_file is not None:
    try:
        product_df = pd.read_csv(upload_file)
    except Exception as e:
        st.error(f"Error: {e}")

    st.subheader("logisticreg_data")
    st.write(product_df.head())

    features = ['Age', 'EstimatedSalary']
    target = 'Purchased'

    X = product_df[features]
    y = product_df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=0)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    st.subheader("Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(x='EstimatedSalary', y='Purchased', data=product_df, hue='Purchased', ax=ax)
    plt.title("Purchase Decision vs Salary")
    st.pyplot(fig)
