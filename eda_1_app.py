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
     page_title='EDA App1',
     page_icon=':tada:',
     layout='wide',
     initial_sidebar_state='expanded')

#--------------------------------
# ASSETS
#--------------------------------
testdata = pd.read_csv("data/breast_cancer.csv")
diamonds = pd.read_csv("data/preprocessed_diamonds.csv")
# clean = Image.open("img/ml.png")
cover = Image.open("img/diamonds.png")
logo = Image.open("img/logo.png")
weather = pd.read_csv('data/seattle_weather.csv')

#--------------------------------
# PAGE HEADER
#--------------------------------
def header():
  with st.container():  
    st.markdown("<h1 style='text-align: center; color: #AAAAAA; font-size: 45px;'>Basic Exploratory Data Analysis Web Applications</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #BBBBBB; font-size: 35px;'> Build Using Python and Streamlit Library</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #CCCCCC; font-size: 20px;'> Developer: Teresia Mrema Buza</h2>", unsafe_allow_html=True)
    st.markdown("<center><a href='https://complexdatainsights.com'><img src='https://complexdatainsights.com/wp-content/uploads/2020/10/logo-1.png' alt=' page cover' width='20%' style='padding: 0 15px; float: center;'/></a></center>", unsafe_allow_html=True)
  
  st.write("##")
  st.write("##")
  
  with st.container():   
    filePath = './eda_1_app.py'
    modTimesinceEpoc = os.path.getmtime(filePath)
    modificationTime = datetime.datetime.fromtimestamp(modTimesinceEpoc).strftime('%Y-%m-%d %H:%M:%S')
    st.write("Last Modified: ", modificationTime )

if header():
  header()

def main():
  with st.container():
    st.success(
    """ 
    # :shower: EDA 1: Dataframe Exploration
    """)
    
    col1, col2 = st.columns((1, 2))
    with col1:
      st.markdown(
        """
        ### App Functionalities
        - This App helps you to create a tidy dataframe for downstream machine learning problems.
          - Reads the `data` uploaded by user and automatically starts exploration.
          - Displays the `head` and `tail` of the uploaded data (input dataframe).
          - Shows the dataframe `dimension`, `variable names` and `missing values`.
          - Perform `descriptive` statistical analysis of the numerical variables.
        """)
        
      uploaded_file = st.sidebar.file_uploader("Please choose a CSV file", type=["csv"])
      if uploaded_file is not None:
        def load_data():
          a = pd.read_csv(uploaded_file)
          return a
        
        with col2:
          df = load_data()
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
        
        with col2:
          df = load_data()
          st.subheader("""Data Visualization""")
             
if __name__ == '__main__':
  main()
  
st.write("---")
st.write("##")
st.write("##")


