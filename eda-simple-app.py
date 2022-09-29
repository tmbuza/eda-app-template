import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Page configuration
st.set_page_config(
     page_title='Exploratory Analysis App',
     page_icon=':tada:',
     layout='wide',
     initial_sidebar_state='expanded')

# App Title
st.title("EDA App")
st.markdown("""
## An Exploratory Data Analysis App
> This EDA App is a modified version of an original app by [Chanin Nantasenamat](https://medium.com/@chanin.nantasenamat) (aka [Data Professor](http://youtube.com/dataprofessor)). Feel free to improve it further.
""")

with st.container():
  st.write("##")
  st.write("---")
  st.header("Data Preprocessing")
  col1, separator1, col2, separator2, col3 = st.columns((1, 0.2, 1, 0.2, 1)) 
  
  with col1:
    st.subheader("Import input data")
    df = pd.read_csv('data/jmwt.csv')
    st.dataframe(df)

  with col2:
    st.subheader("With transformed Date")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    st.dataframe(df)
    df.to_csv('data/jmwt2.csv', index = False)

    
  with col3: 
    st.subheader("Data manipulation")
    st.write("Extract MEK data and add metadata")
    df_mek = df.drop("JOB", axis=1, inplace=False)
    df_mek.columns = ['Weight']
    df_mek['Age'] = 'Below 30'
    df_mek['Occupation'] = 'Data Scientist'
    df_mek['Label'] = 'MEK'
    st.dataframe(df_mek.head(3))

    st.write("Extract JOB data and add metadata")
    df_job = df.drop("MEK", axis=1, inplace=False)
    df_job.columns = ['Weight']
    df_job['Age'] = 'Above 30'
    df_job['Occupation'] = 'Lead Developer'
    df_job['Label'] = 'JOB'
    st.dataframe(df_job.head(3))

with st.container():
  st.write("##")
  st.write("---") 
  st.header("Data Exploration")

# Upload CSV data
with st.sidebar.header('1. Upload Tidy CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/tmbuza/my-app-template/main/data/jmwt2.csv)
""")

# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)
else:
    st.info(':exclamation: Awaiting user\'s input file. Please make sure you uploaded a tidy data using the `user input widget` on the left sidebar.')
    if st.button('Example data'):
        # Example data
        @st.cache
        def load_data():
            a = pd.read_csv('data/jmwt2.csv')
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
