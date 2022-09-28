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
    st.subheader("Import weight data")
    df = pd.read_csv('data/jmwt.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(['Date'])
    df.to_csv('data/jmwt2.csv', index=True)
    # st.dataframe(df)
    
# Upload CSV data
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
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
    st.info('Awaiting user\'s input file')
    if st.button('Example data'):
        # Example data
        @st.cache
        def load_data():
            a = pd.DataFrame(
                np.random.rand(100, 5),
                columns=['a', 'b', 'c', 'd', 'e']
            )
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
