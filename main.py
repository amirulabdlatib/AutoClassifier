import streamlit as st
import pandas as pd
import os
import ydata_profiling as ydp
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models, pull as classification_pull, save_model as classification_save_model


choices = ['Upload','Profiling','Machine Learning Modelling']

if os.path.exists('source_data.csv'):
    df = pd.read_csv('source_data.csv')
    
    
with st.sidebar:
    st.title('Automl')
    choice = st.radio(label='Navigation',options=choices)
    st.info('This application allows you to build an automated ML pipeline using Streamlit, Pandas Proflling and PyCaret')


if choice == 'Upload':
    file = st.file_uploader('Upload your file in CSV Here')
    if file:
        df = pd.read_csv(file)
        df.to_csv('source_data.csv',index=None)
        st.dataframe(df)
        

if choice == 'Profiling':
    st.title('Auto Profiling | Exploratory Data Analysis')
    try:
        profile_report = ydp.ProfileReport(df)
        st_profile_report(profile_report)
    except AttributeError as e:
        st.warning(e)

if choice == 'Machine Learning Modelling':
    st.title('Machine Learning')
    target = st.selectbox(label='Select Your Target',options=df.columns)
    if st.button('Train Model'):

        try:
            classification_setup(df,target=target)
            setup_df = classification_pull()
            st.info('This is the ML Classification Experiment settings')
            st.dataframe(setup_df)

            best_model = classification_compare_models()
            compare_df = classification_pull()
            st.info('This is the ML Models')
            st.dataframe(compare_df)
            classification_save_model(best_model,'best_model')

        
            if os.path.exists('best_model.pkl'):
                with open('best_model.pkl', 'rb') as f:
                    model_bytes = f.read()

            st.download_button(label='Download the model', data=model_bytes, file_name='best_model.pkl')
        
        except Exception as e:
            st.error(e)