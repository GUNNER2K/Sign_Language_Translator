import streamlit as st

from scripts.capture_script import *

from tensorflow.keras.models import load_model



# @st.cache_resource
# def load_model():
#     model = load_model('App/assets/asl_model_2.h5')
#     return model

st.set_page_config(layout='wide')

# def update_word():
#     global word
#     return word


def init_translation():
    st.session_state['start_translation'] = True
    st.session_state['close_camera'] = False
 
def close_translation():
    st.session_state['close_camera'] = True

st.header('ASL Translator App')

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
        st.write('''The Dataset we are using is an image dataset consisting of around 2000 images across 36 different classes ranging from A-Z and 0-9.
                 
                 Dataset Link: https://www.kaggle.com/datasets/ayuraj/asl-dataset
                 
                 
This Dataset was preprocessed and Augmented and was used to train our model on.''')

    with col2:
        st.subheader("Dataset Sample Images")
        imga = cv2.imread(r'App\assets\example_ds.png')
        imga = cv2.cvtColor(imga, cv2.COLOR_BGR2RGB)
        imgb = cv2.imread(r'App\assets\class_distribution.png')
        imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2RGB)
        st.image(imga)
        st.image(imgb)

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
        if 'start_translation' not in st.session_state:
            st.session_state['start_translation'] = False
        if 'close_camera' not in st.session_state:
            st.session_state['close_camera'] = False
        if 'word' not in st.session_state:
            st.session_state['word'] = 'Krish L Sharma'
        st.button('Start Translator', on_click= init_translation)
        st.button('Stop', on_click= close_translation)
        # print(st.session_state)
        if st.session_state['start_translation']:
            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            model = load_model(model_path)
            frame_counter = 0
            while cap.isOpened():
                # stop = st.session_state.stop_button
                frame_counter += 1
                ret, frame = cap.read()

                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frame , coord = draw_hands(frame)
                if frame_counter == 20:
                    if coord:
                        letter= draw_prediction(frame, coord, model)
                        word = word+ letter
                        st.session_state['word'] += letter
                        frame_counter = 0
                    else:
                        frame_counter = 0
                        if word != '':
                            st.session_state['word'] += ' '
                            word = ''
                if not ret:
                    st.write('Video Capture has ended.')
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_placeholder.image(frame, channels='BGR')
                
                # print(stop)
                # print(st.session_state)
                if st.session_state['close_camera']:
                    # print('Stopping the translator')
                    cap.release()
                    cv2.destroyAllWindows()
                    frame_placeholder.empty()
                    st.session_state['start_translation'] = False
                    break
                #st.write(prediction)
            
            # print('Clearing the column')

        st.empty()

    with col2:
        st.subheader('Prediction')
        show_word = st.session_state['word']
        st.empty()
        st.write(show_word)
        st.session_state['word'] = ''
        