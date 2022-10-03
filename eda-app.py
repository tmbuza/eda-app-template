import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import plotly.figure_factory as ff
import seaborn as sns
        
# Page configuration
st.set_page_config(
     page_title='Exploratory Analysis App',
     page_icon=':tada:',
     layout='wide',
     initial_sidebar_state='expanded')

# App Title
st.title("EDAA: Exploratory Data Analysis App")
st.markdown(
  """
  #### EDAA does the following:
  - Reads the data uploaded by user.
  - Displays few line of the uploaded data.

  #### Users must:
  - Upload input file using the user\'s widget on the left sidebar. 
  - Make sure that the uploaded data is tidy. See the input data structure [here]().
  """
)

with st.sidebar.header('User input widget'):
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    st.sidebar.markdown("""
""")
# [Example CSV input file](https://raw.githubusercontent.com/tmbuza/my-app-template/main/data/iris.csv)
if uploaded_file is None:
    st.info(':exclamation: Awaiting user\'s input file')
    
    if st.button('Run a Demo Data'):
      # Using a demo data
      @st.cache
      def load_data():
          a = pd.read_csv('data/iris.csv')
          return a
      df = load_data()
      st.header("""Demo Data Exploration""")
      st.subheader("""Input DataFrame""")
      st.write("Head", df.head())
      st.write("Tail", df.tail())
  
      st.subheader("""Dataframe dimension""")
      st.markdown("> Note that 0 = rows, and 1 = columns")
      st.dataframe(df.shape)
  
      st.subheader("""Variable names""")
      st.dataframe(pd.DataFrame({'Variable': df.columns}))
      
      st.subheader("""Missing values""")
      missing_count = df.isnull().sum()
      value_count = df.isnull().count() #denominator
      missing_percentage = round(missing_count/value_count*100, 2)
      missing_df = pd.DataFrame({'Count': missing_count, 'Missing (%)': missing_percentage})
      st.dataframe(missing_df)
   
      st.subheader("""Descriptive statistics""")
      st.dataframe(df.describe())
  
      st.subheader("""Correlation heatmap""")    
      fig, ax = plt.subplots()
      sns.heatmap(df.corr(), ax = ax)
      st.write(fig, use_container_width=False) 
    
else:
    @st.cache
    def load_data():
        a = pd.read_csv(uploaded_file)
        return a
    df = load_data()
    st.header("""Data Exploration""")
    st.subheader("""Input DataFrame""")
    st.write("Head", df.head())
    st.write("Tail", df.tail())

    st.subheader("""Dataframe dimension""")
    st.markdown("> Note that 0 = rows, and 1 = columns")
    st.dataframe(df.shape)

    st.subheader("""Variable names""")
    st.dataframe(pd.DataFrame({'Variable': df.columns}))
    
    st.subheader("""Missing values""")
    missing_count = df.isnull().sum()
    value_count = df.isnull().count() #denominator
    missing_percentage = round(missing_count/value_count*100, 2)
    missing_df = pd.DataFrame({'Count': missing_count, 'Missing (%)': missing_percentage})
    st.dataframe(missing_df)

    st.header("""Basic Statistics""")    
    st.subheader("""Descriptive statistics""")
    st.dataframe(df.describe())

    st.subheader("""Correlation heatmap""")    
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax = ax)
    st.write(fig, use_container_width=False) 
