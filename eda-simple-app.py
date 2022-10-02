import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import plotly.figure_factory as ff
        
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

# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    st.header('**Input DataFrame**')
    st.write(df)
    # st.header('**Processed DataFrame**')
    # st.write(df)
    # st.write('---')
    # st.header('**Pandas Profiling Report**')
    # pr = ProfileReport(df, explorative=False)
    # st_profile_report(pr)
    st.subheader("Descriptive statistics")
# st.markdown("Categorical variables")
# st.dataframe(df.describe(include='all'))

    st.markdown("Numerical variables")
    st.dataframe(df.describe())
    
    st.subheader("Count distinct elements")
    st.write("""
    - Typically: 0 = rows [Default], 1 = headers or columns
    """)
    st.dataframe(df.nunique(axis=0))
    
    st.subheader("Plotly bar charts")
    st.write("""
    - Here `plotly` uses a `pandas` dataframe.
    - Pandas uses `matplotlib` to create the plot.
    """)
    df = df.set_index(df.columns[0])
    st.write(df)
    st.bar_chart(df, use_container_width=True)
    
    x = df.index
    y1 = df[df.columns[0]]
    y2 = df[df.columns[1]]
    
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    
    ax1.plot(x, y1, color='tab:pink', lw=1)
    ax2.plot(x, y2, color='tab:blue', lw=1)
    
    ax1.set_ylabel("MEK", color='tab:pink', fontsize=8)
    ax1.tick_params(axis="y", labelcolor='tab:pink')
    
    ax2.set_ylabel("JOB", color='tab:blue', fontsize=8)
    ax2.tick_params(axis="y", labelcolor='tab:blue')
    
    fig.suptitle("MEK & JOB Weight Tracking", fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()
    
    st.pyplot(fig)

else:
    st.info(':exclamation: Awaiting user\'s input file.')
    # if st.button('Example data'):
    #     # Example data
    #     @st.cache
    #     def load_data():
    #         a = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
    #         return a
    #     df = load_data()
    #     st.header('**Input DataFrame**')
    #     st.write(df)
    #     st.write('---')
    #     st.header('**Data Exploration Report**')
    # 
    #     # x = df['Date']
    #     # y1 = df['MEK']
    #     # y2 = df['JOB']
    #     # 
    #     # fig, ax1 = plt.subplots(figsize=(8, 8))
    #     # ax2 = ax1.twinx()
    #     # 
    #     # ax1.plot(x, y1, color='tab:pink', lw=1)
    #     # ax2.plot(x, y2, color='tab:blue', lw=1)
    #     # 
    #     # ax1.set_ylabel("MEK", color='tab:pink', fontsize=8)
    #     # ax1.tick_params(axis="y", labelcolor='tab:pink')
    #     # 
    #     # ax2.set_ylabel("JOB", color='tab:blue', fontsize=8)
    #     # ax2.tick_params(axis="y", labelcolor='tab:blue')
    #     # 
    #     # fig.suptitle("MEK & JOB Weight Tracking", fontsize=10)
    #     # fig.autofmt_xdate()
    #     # fig.tight_layout()
    #     # plt.show()
    #     # 
    #     # st.pyplot(fig)
    #     st.subheader("Descriptive statistics")
    # # st.markdown("Categorical variables")
    # # st.dataframe(df.describe(include='all'))
    # 
    #     st.markdown("Numerical variables")
    #     st.dataframe(df.describe())
    #     
    #     st.subheader("Count distinct elements")
    #     st.write("""
    #     - Typically: 0 = rows [Default], 1 = headers or columns
    #     """)
    #     st.dataframe(df.nunique(axis=0))
    #     
    #     st.subheader("Plotly bar charts")
    #     st.write("""
    #     - Here `plotly` uses a `pandas` dataframe.
    #     - Pandas uses `matplotlib` to create the plot.
    #     """)
    #     df = df.groupby(by=["day"]).sum()[["tip"]].sort_values(by=["tip"])
    #     df = df.set_index(df.columns[1])
    #     st.write(df)        
    #     st.bar_chart(df, use_container_width=True)  

