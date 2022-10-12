import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.core.numeric import True_
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import plotly.express as px
import os
import stat
import time
import datetime

#--------------------------------
# PAGE CONFIG
#--------------------------------
st.set_page_config(
     page_title='WebApps',
     page_icon=':tada:',
     layout='wide',
     initial_sidebar_state='expanded')

#--------------------------------
# ASSETS
#--------------------------------
testdata = pd.read_csv("data/breast_cancer.csv")
diamonds = pd.read_csv("data/preprocessed_diamonds.csv")
clean = Image.open("img/ml.png")
cover = Image.open("img/diamonds.png")
logo = Image.open("img/logo.png")
weather = pd.read_csv('data/seattle_weather.csv')

#--------------------------------
# PAGE HEADER
#--------------------------------
def header():
  with st.container():  
    st.markdown("<h1 style='text-align: center; color: #AAAAAA; font-size: 60px;'>Exploratory Data Analysis Web Applications</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #CCCCCC; font-size: 45px;'> Build Using Python and Streamlit Library</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #CCCCCC; font-size: 20px;'> Author: Teresia Mrema Buza</h2>", unsafe_allow_html=True)
    st.markdown("<center><a href='https://complexdatainsights.com'><img src='https://complexdatainsights.com/wp-content/uploads/2020/10/logo-1.png' alt=' page cover' width='10%' style='padding: 0 15px; float: center;'/></a></center>", unsafe_allow_html=True)
  
  st.write("##")
  st.write("##")
  
  with st.container():   
    filePath = './eda-app.py'
    modTimesinceEpoc = os.path.getmtime(filePath)
    modificationTime = datetime.datetime.fromtimestamp(modTimesinceEpoc).strftime('%Y-%m-%d %H:%M:%S')
    st.write("Last Modified: ", modificationTime )

if header():
  header()

def main():
  with st.container():
    st.success(
    """ 
    # :shower: Data Exploration App
    """)
    
    col1, col2 = st.columns(2)
    with col1:
      st.image(cover, width=450)        
    with col2:
      st.markdown(
        """
        ### App Functionalities
        - This App helps you to create a tidy dataframe for downstream machine learning problems.
          - Reads the `data` uploaded by user and automatically starts exploration.
          - Displays the `head` and `tail` of the uploaded data (input dataframe).
          - Shows the dataframe `dimension`, `variable names` and `missing values`.
          - Perform `descriptive` statistical analysis of the numerical variables.
          - Plots a `correlation` heatmap.
          - Create Plotly interactive plots
        """)
  
      st.write("""### Testing the App""")
      st.markdown(
        """
        - You may start by testing the app using a demo data by clicking on the `Run a Partial Demo` button, otherwise:
        - Upload a tidy input file using the user\'s widget on the left sidebar.
        """)
      if st.button('Run a Partial Demo'):
        # Using a demo data
        @st.cache
        def load_data():
            a = pd.read_csv('data/preprocessed_diamonds.csv')
            return a
        df = load_data()
        st.write(""" > This demo uses a preprocessed `diamond dataset` for demonstration only.""")
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
      
        st.header("""Basic Statistics""")
        st.subheader("""Descriptive statistics""")
        st.dataframe(df.describe())
  
        st.write("##")
        st.write("##")
        st.subheader("""Correlation heatmap""")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), ax = ax)
        st.write(fig, use_container_width=False)

if __name__ == '__main__':
  main()
  
st.write("---")


# import numpy as np
# import pandas as pd
# import streamlit as st
# from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
# 
# import plotly.figure_factory as ff
# import seaborn as sns
# 
#        
# # Page configuration
# st.set_page_config(
#      page_title='EDA App',
#      page_icon=':bar_chart:',
#      layout='centered',
#      initial_sidebar_state='expanded')
#      
# from plotnine.data import diamonds
# 
# # diamonds = diamonds.iloc[:, 0:]
# # diamonds.to_csv("data/preprocessed_diamonds.csv", index = False) 
# # iris = sns.load_dataset('iris')
# # 
# # # App Title
# # with st.container():
# #   st.write("##")
# #   st.title("EDAA: Exploratory Data Analysis App")
# # 
# #   panel1, panel2 = st.columns((2))
# #   with panel1:
# #     st.markdown(
# #       """
# #       #### The EDAA does the following:
# #       - Reads the `data` uploaded by user and automatically starts exploration.
# #       - Displays the `head` and `tail` of the uploaded data (input dataframe).
# #       - Shows the dataframe `dimension`, `variable names` and `missing values`.
# #       - Perform `descriptive` statistical analysis of the numerical variables.
# #       - Plots a `correlation` heatmap.
# #       """)
# #       
# #   with panel2:
# #     st.info(
# #       """
# #       #### Users must:
# #       - Upload input file using the user\'s widget on the left sidebar. 
# #       - Make sure that the uploaded data is tidy.
# #       """
# #     )
# # 
# with st.sidebar.header('User input widget'):
#     uploaded_file = st.sidebar.file_uploader("Please choose a tidy CSV file", type=["csv"])
#
# if uploaded_file is None:
#     st.warning(':exclamation: Awaiting user\'s input file')
#
#     if st.button('Run a Demo Data'):
#       # Using a demo data
#       @st.cache
#       def load_data():
#           a = pd.read_csv('data/preprocessed_diamonds.csv')
#           return a
#       df = load_data()
#       st.write(""" > This demo uses a preprocessed `diamond dataset` for demonstration only.""")
#       st.header("""Demo Data Exploration""")
#       st.subheader("""Input DataFrame""")
#       st.write("Head", df.head())
#       st.write("Tail", df.tail())
#
#       st.subheader("""Dataframe dimension""")
#       st.markdown("> Note that 0 = rows, and 1 = columns")
#       st.dataframe(df.shape)
#
#       st.subheader("""Variable names""")
#       st.dataframe(pd.DataFrame({'Variable': df.columns}))
#
#       st.subheader("""Missing values""")
#       missing_count = df.isnull().sum()
#       value_count = df.isnull().count() #denominator
#       missing_percentage = round(missing_count/value_count*100, 2)
#       missing_df = pd.DataFrame({'Count': missing_count, 'Missing (%)': missing_percentage})
#       st.dataframe(missing_df)
#
#       st.subheader("""Descriptive statistics""")
#       st.dataframe(df.describe())
#
#       st.subheader("""Correlation heatmap""")
#       fig, ax = plt.subplots()
#       sns.heatmap(df.corr(), ax = ax)
#       st.write(fig, use_container_width=False)
#
# else:
#     @st.cache
#     def load_data():
#         a = pd.read_csv(uploaded_file)
#         return a
#     df = load_data()
#     st.header("""Data Exploration""")
#     st.subheader("""Input DataFrame""")
#     st.write("Head", df.head())
#     st.write("Tail", df.tail())
#
#     st.subheader("""Dataframe dimension""")
#     st.markdown("> Note that 0 = rows, and 1 = columns")
#     st.dataframe(df.shape)
#
#     st.subheader("""Variable names""")
#     st.dataframe(pd.DataFrame({'Variable': df.columns}))
#
#     st.subheader("""Missing values""")
#     missing_count = df.isnull().sum()
#     value_count = df.isnull().count() #denominator
#     missing_percentage = round(missing_count/value_count*100, 2)
#     missing_df = pd.DataFrame({'Count': missing_count, 'Missing (%)': missing_percentage})
#     st.dataframe(missing_df)
#
#     st.header("""Basic Statistics""")
#     st.subheader("""Descriptive statistics""")
#     st.dataframe(df.describe())
#
#     st.subheader("""Scatter plot""")
#     fig = sns.FacetGrid(iris, hue ="species",
#                   height = 6).map(plt.scatter,'sepal_length', 'petal_length').add_legend()
#     st.pyplot(fig)
#
#
#     st.subheader("""Correlation heatmap""")
#     fig, ax = plt.subplots()
#     sns.heatmap(df.corr(), ax = ax)
#     st.write(fig, use_container_width=False)


# diamonds = diamonds.iloc[:, 0:]
# diamonds.to_csv("data/preprocessed_diamonds.csv", index = False)
# 
# with st.container():
#   st.success(
#   """
#   # Exploratory Data Analysis App
#   """)
# # panel1, panel2, panel3 = st.columns((2, 0.12, 2))
# # with panel1:
#   st.markdown(
#     """
#     ### App Functionalities
#     - Reads the `data` uploaded by user and automatically starts exploration.
#     - Displays the `head` and `tail` of the uploaded data (input dataframe).
#     - Shows the dataframe `dimension`, `variable names` and `missing values`.
#     - Perform `descriptive` statistical analysis of the numerical variables.
#     - Plots a `correlation` heatmap.
#     """)
# 
# 
# # with panel3:
#   st.write("""
#     ### Testing the App""")
#   st.markdown(
#     """
#     - Upload input file using the provided user\'s widget.
#     - Make sure that the uploaded data is tidy.
#     - You may start by testing the app using a demo data.
#     - To test the app click on the `Test EDA App` button.
#     """)
# 
#   if st.button('Click to Test the App'):
#     # Using a demo data
#     @st.cache
#     def load_data():
#         a = pd.read_csv('data/preprocessed_diamonds.csv')
#         return a
# 
#     df = load_data()
#     st.write(""" > This demo uses a preprocessed `diamond dataset` for demonstration only.""")
#     st.header("""Demo Data Exploration""")
#     st.subheader("""Input DataFrame""")
#     st.write("Head", df.head())
#     st.write("Tail", df.tail())
# 
#     st.subheader("""Dataframe dimension""")
#     st.markdown("> Note that 0 = rows, and 1 = columns")
#     st.dataframe(df.shape)
# 
#     st.subheader("""Variable names""")
#     st.dataframe(pd.DataFrame({'Variable': df.columns}))
# 
#     st.subheader("""Missing values""")
#     missing_count = df.isnull().sum()
#     value_count = df.isnull().count() #denominator
#     missing_percentage = round(missing_count/value_count*100, 2)
#     missing_df = pd.DataFrame({'Count': missing_count, 'Missing (%)': missing_percentage})
#     st.dataframe(missing_df)
# 
#     st.subheader("""Descriptive statistics""")
#     st.dataframe(df.describe())
# 
#     st.subheader("""Correlation heatmap""")
#     fig, ax = plt.subplots()
#     sns.heatmap(df.corr(), ax = ax)
#     st.write(fig, use_container_width=False)
# 
# st.write("---")
# st.write("##")
# 
with st.container():
#   # panel1, panel2, panel3 = st.columns((0.3, 1, 0.3))
#   #
#   # with panel2:
#   st.write("""
#     ### User input widget""")
#   st.markdown(
#     """ """)
  uploaded_file = st.sidebar.file_uploader("Please choose a CSV file", type=["csv"])
  if uploaded_file is not None:
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

    st.header("""Basic Visualization""")
    st.subheader("""Line charts""")
    st.line_chart(df, x="date", y="temp_min")
    st.line_chart(df, x="date", y="temp_max")
    st.line_chart(df, x="date", y="wind")
    
    st.subheader("""Bar charts""")
    st.bar_chart(df, x="date", y="temp_min")
    st.bar_chart(df, x="date", y="temp_max")
    st.bar_chart(df, x="date", y="wind")
    
    st.subheader("""Area charts""")
    st.area_chart(df, x="date", y="temp_min")
    st.area_chart(df, x="date", y="temp_max")
    st.area_chart(df, x="date", y="wind")
    
    st.line_chart(
    df,
    x="date",
    y=["temp_min", "temp_max", "wind"],  # <-- You can pass multiple columns!
    )
    
    st.subheader("""Correlation heatmap""")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax = ax)
    st.write(fig, use_container_width=False)
    
  else:
    st.warning(':exclamation: Awaiting user\'s input file')

st.write("##")
st.write("##")


