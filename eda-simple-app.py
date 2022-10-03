import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import skimpy
from skimpy import skim, generate_test_data
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
st.markdown("""
#### This application does the following:
- Reads the data uploaded by user.
- Displays few line of the uploaded data.

""")
# 
# with st.container():
#   st.write("##")
#   st.write("---")
#   st.header("Data Preprocessing")
#   col1, separator1, col2, separator2, col3 = st.columns((1, 0.2, 1, 0.2, 1)) 
#   
#   with col1:
#     st.subheader("Import input data")
#     df = pd.read_csv('data/jmwt.csv')
#     st.dataframe(df)
# 
#   with col2:
#     st.subheader("With transformed Date")
#     df['Date'] = pd.to_datetime(df['Date'])
#     # df = df.set_index('Date')
#     st.dataframe(df)
#     df.to_csv('data/jmwt2.csv', index = False)

    
  # with col3: 
  #   st.subheader("Data manipulation")
  #   st.write("Extract MEK data and add metadata")
  #   df_mek = df.drop("JOB", axis=1, inplace=False)
  #   df_mek.columns = ['Weight']
  #   df_mek['Age'] = 'Below 30'
  #   df_mek['Occupation'] = 'Data Scientist'
  #   df_mek['Label'] = 'MEK'
  #   st.dataframe(df_mek.head(3))
  # 
  #   st.write("Extract JOB data and add metadata")
  #   df_job = df.drop("MEK", axis=1, inplace=False)
  #   df_job.columns = ['Weight']
  #   df_job['Age'] = 'Above 30'
  #   df_job['Occupation'] = 'Lead Developer'
  #   df_job['Label'] = 'JOB'
  #   st.dataframe(df_job.head(3))

with st.sidebar.header('Prerequisites'):
  st.header("Start by entering data")
  st.markdown(
    """
    - Upload input file using the user\'s widget on the left sidebar. 
    - Please make sure you uploaded a tidy data to enable the app work smoothly.
    """
  )
  st.header("Data processing glimpse")
  st.markdown(
    """
    ...in progress...

    """)  
  st.write("---") 


# Upload CSV or XLSX file
with st.sidebar.header('1. Upload Tidy CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/tmbuza/my-app-template/main/data/iris.csv)
""")

# # Pandas Profiling Report
# if uploaded_file is not None:
#     @st.cache
#     def load_csv():
#         csv = pd.read_csv(uploaded_file)
#         return csv
#     df = load_csv()
#     st.header('**Input DataFrame**')
#     st.write(df)
#     st.header('**Pandas Profiling Report**')
#     pr = ProfileReport(df, explorative=False)
#     st_profile_report(pr)

# else:
#     st.info(':exclamation: Awaiting user\'s input file.')

if uploaded_file is not None:
    @st.cache
    def load_df():
        df = pd.read_csv(uploaded_file)
        return df
      
    df = load_df()
    st.header("""Data Exploration""")
    st.subheader("""Input DataFrame""")
    st.write("Head", df.head())
    st.write("Tail", df.tail())

    st.subheader("""Dataframe dimension""")
    st.markdown("> Note that 0 = rows, and 1 = columns")
    st.dataframe(df.shape)

    st.subheader("""Variable names""")
    st.dataframe(pd.DataFrame({'Variable': df.columns}))
    
    st.subheader("""Data information""")
    st.dataframe(df.info())
    
    st.subheader("""Descriptive statistics""")
    st.dataframe(df.describe())
    
    st.subheader("""Missing values""")
    missing_count = df.isnull().sum()
    value_count = df.isnull().count() #denominator
    missing_percentage = round(missing_count/value_count*100, 2)
    missing_df = pd.DataFrame({'Count': missing_count, 'Missing (%)': missing_percentage})
    st.dataframe(missing_df)

    # barchart = missing_df.plot.bar(y='Missing (%)') # Adding labels to the bar using for loop
    # fig, ax = plt.subplots()
    # for index, percentage in enumerate(missing_percentage):
    #     barchart.text(index, percentage, str(percentage) + '%')
    # st.pyplot(fig) 
    # 
    # fig, ax = plt.subplots()    
    # plt.figure(figsize=(7, 7))
    # sns.heatmap(df.isnull(), yticklabels=False, cmap='viridis')
    # st.write(fig)   

    
    st.header("""Basic Statistics""")    
    st.subheader("""Categorical variable""")    
    for col in df.columns:
      if df[col].dtype == 'object':
        st.write(df[col].value_counts())

    st.subheader("""Correlation heatmap""")    
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax = ax)
    st.write(fig, use_container_width=False) 
    
    # st.write(skim(df))

else:
    st.info(':exclamation: Awaiting user\'s input file.')
