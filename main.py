import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title='Simple Prediction App',
    layout='wide',
    initial_sidebar_state='auto',)

# Connect Kaggle with Streamlit

# Initialize Kaggle API client and authenticate using secrets
def download_dataset():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.set_config_value('username', st.secrets["kaggle"]["username"])
    api.set_config_value("key", st.secrets["kaggle"]["key"])
    api.authenticate()
    # Add dataset
    dataset = "mlg-ulb/creditcardfraud"
    path = "data"
    # Download dataset
    api.dataset_download_files(dataset, path=path, unzip=True)



# # Load the image
img = Image.open('img/overview_photo.jpg')

# # convert it to base64
buffered = BytesIO()
img.save(buffered, "JPEG")
img_bs64 = base64.b64encode(buffered.getvalue()).decode()

body = f"""
<div style="text-align: center;">
    <img src="data:img/jpeg;base64,{img_bs64}" width="250" height="200"/>
</div>
"""
st.markdown(body, unsafe_allow_html=True)

# Title of the application
st.title('🔓Credit Card Fraud Detection App') 
credit_card_df = pd.read_csv("data/creditcard.csv")
st.write(credit_card_df) 

# # Data Overview
# st.subheader("📊 Data Overview")

# # View first 5 rows
# st.dataframe(data=credit_card_df.head(6), height=200, width=2500)

# # View last 5 rows
# st.dataframe(data=credit_card_df.tail(5), height=200, width=2500)

# st.subheader("Credict Card Fraud EDA")
# st.dataframe(credit_card_df.describe(), height=200, width=2500)
# st.dataframe(credit_card_df.info(), height=200, width=2500)
# st.dataframe(credit_card_df.shape)
# st.dataframe(credit_card_df.count(), height=200, width=2500)

# # Rename some columns
# credit_card_df = credit_card_df.rename(columns={'Amount': 'Transaction_Amount', 'Class': 'Is_Fraudulent'})
# st.dataframe(credit_card_df.columns)

# # Find null values
# st.dataframe(credit_card_df.isna().sum(), height=500, width=200)
# print(credit_card_df.shape)

# # Create Correlation Matrix
# # st.title("Correlation Matrix")
# # plt.figure(figsize=(12, 10))
# # sns.heatmap(credit_card_df.corr(), annot=True, smap='Reds')
# # plt.title('Correlation Matrix')
# # st.pyplot(plt)

# # Machine Learning - Baseline Model
# st.subheader("Machine Learning Model")
# st.write("I built Logistic Regression model as a baseline model!")
# X = credit_card_df.drop(['Is_Fraudulent'], axis=1)
# y = credit_card_df['Is_Fraudulent'].copy()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# num_features = X_train.select_dtypes(include=[np.number]).columns
# cat_features = X_train.select_dtypes(exclude=[np.number]).columns
# num_pipeline = Pipeline([
#     ('impute', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# cat_pipeline = Pipeline([
#     ('impute', SimpleImputer(strategy='most_frequent')),
#     ('scaler', StandardScaler())
# ])

# transformer = ColumnTransformer([
#     ('num', num_pipeline, num_features),
#     ('object', cat_pipeline, cat_features)
# ], remainder='passthrough')

# print(transformer.fit(X_train))

# X_train_transformer = pd.DataFrame(data=transformer.transform(X_train), columns=transformer.get_feature_names_out())
# X_test_transformer = pd.DataFrame(data=transformer.transform(X_test), columns=transformer.get_feature_names_out())
# print(X_train_transformer, X_test_transformer)

# st.write("Training and testng scores are listed below:")
# log_reg = LogisticRegression()
# print(log_reg.fit(X_train_transformer, y_train))
# st.write(f"✅Train score: ,{log_reg.score(X_train_transformer, y_train)}")
# print(f"✅Test score: ,{log_reg.score(X_test_transformer, y_test)}")

# st.write("Predictions")
# y_pred = log_reg.predict(X_test_transformer)
# st.bar_chart(data=y_pred,color='#c2522d')

# # Evaluate the performance using metrics
# st.subheader("Evaluation Metrics")
# st.write(f"Accuracy: {accuracy_score(y_pred, y_test)}")
# st.write(f"Precision: {precision_score(y_pred, y_test)}")
# st.write(f"Recall: {recall_score(y_pred, y_test)}")
# st.write(f"F1 score: {f1_score(y_pred, y_test)}")
# st.write(f"ROC-AUC score: {roc_auc_score(y_pred, y_test)}")
