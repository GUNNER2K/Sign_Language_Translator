import streamlit as st

from scripts.capture_script import *

from tensorflow.keras.models import load_model



# @st.cache_resource
# def load_model():
#     model = load_model('App/assets/asl_model_2.h5')
#     return model

st.set_page_config(layout='wide')



st.header('ASL Trasnlator App')

tab1, tab2, tab3, tab4 = st.tabs(['Home', 'Dataset', 'Model', 'Translator'])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('About the App')
        st.write('''This app is a Sign Language to text translator app for people who dont know how to use sign language.
             I'm using the American Sign Language standards for this purpose as it is the most widely used sign language standard out there.''')
        st.write('The App uses a Convolutional Neural Network(CNN) built and trained with the help of Tensorflow to detect and recognize the alphabet that is being spelled by hand.')
        st.markdown('**Techstack used for the app:**')
        st.markdown(' - Tensorflow')
        st.markdown(' - Pandas')
        st.markdown(' - Numpy')
        st.markdown(' - Matplotlib')

    
    with col2:
        st.image('App/assets/asl.jpg')

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Description")

    with col2:
        st.subheader("Dataset Sample Images")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Description")

    with col2:
        st.subheader("Model Architecture")

with tab4:
    # model_path = 'App/assets/asl_model_2.h5'
    # model = load_model(model_path)

    col1, col2 = st.columns(2)
    word = ''
    with col1:
        if st.button('Start Translator', key= 'start_button'):
            frame_placeholder = st.empty()
            translator(frame_placeholder, st)        
    with col2:
        st.write(word)    
        