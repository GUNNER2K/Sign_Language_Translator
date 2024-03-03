import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector

from scripts.capture_script import *

from tensorflow.keras.models import load_model


@st.cache_resource
def load_model():
    model = tf.keras.saving.load_model(
        'Models/asl_model_1.h5', custom_objects=None, compile=False, safe_mode=True
    )
    return model

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


with tab4:
    model = load_model()

    col1, col2 = st.columns(2)
    with col1:
        press = st.button('Start Translator')
        frame_placeholder = st.empty()
        stop = st.button('Stop Translate')
        if press:
            cap = cv2.VideoCapture(0)
        
            while cap.isOpened() and not stop:
                ret, image = cap.read()
            
                new_img, image = hand_capture(image)
                img = image.img_to_array(new_img)
                img = np.expand_dims(img, axis=0)
                prediction = model.predict(img)
                prediction = np.argmax(prediction,axis=1)
                print(prediction)
                #prediction = model.predict(crop_img)
                if not ret:
                    st.write('Video Capture has ended.')
                    break
                frame_placeholder.image(image, channels='BGR')
                #st.write(prediction)
                if cv2.waitKey(1) & 0xFF == ord('q') or stop:
                    break
            cap.release()
            cv2.destroyAllWindows()
    # with col2:
    #     st.write(prediction)    
        