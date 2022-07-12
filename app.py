from distutils.command.upload import upload
import pickle
import streamlit as st
import pandas as pd
import numpy as np

st.title('MSCI 436 Final Project')
st.header("Detecting Fradulent Transactions")

@st.cache(allow_output_mutation=True)
def load_model():
    return pickle.load(open('model.pickle.dat', 'rb'))

model = load_model()
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)