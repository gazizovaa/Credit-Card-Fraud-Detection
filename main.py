import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import streamlit as st
from PIL import Image


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


if st.sidebar.button('Get Data', type='primary'):
    download_dataset()

credit_card_df = pd.read_csv("data/creditcard.csv")

# load the image
img = Image.open('img/photo.jpg')

# convert it to base64
buffered = BytesIO()
img.save(buffered, "JPEG")
img_bs64 = base64.b64encode(buffered.getvalue()).decode()

body = f"""
<div style="text-align: center;">
    <img src="data:img/jpeg;base64,{img_bs64}" width="250" height="200"/>
</div>
"""
st.markdown(body, unsafe_allow_html=True)
st.markdown("# 🔓Credit Card Fraud Detection")

# Data Overview
st.subheader("📊 Data Overview")
st.dataframe(data=credit_card_df.head(6), height=200, width=2500)
