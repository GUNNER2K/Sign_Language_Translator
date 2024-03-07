import mediapipe as mp
import cv2
import tensorflow as tf
import streamlit as st

mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
detector = mp.solutions.hands.Hands(static_image_mode= False, min_detection_confidence= 0.8, min_tracking_confidence= 0.5, max_num_hands= 2)
model_path = 'App/assets/asl_model_2.h5'

categories = {  0: "0",
                1: "1",
                2: "2",
                3: "3",
                4: "4",
                5: "5",
                6: "6",
                7: "7",
                8: "8",
                9: "9",
                10: "a",
                11: "b",
                12: "c",
                13: "d",
                14: "e",
                15: "f",
                16: "g",
                17: "h",
                18: "i",
                19: "j",
                20: "k",
                21: "l",
                22: "m",
                23: "n",
                24: "o",
                25: "p",
                26: "q",
                27: "r",
                28: "s",
                29: "t",
                30: "u",
                31: "v",
                32: "w",
                33: "x",
                34: "y",
                35: "z",
            }

@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def calcBoundaryBox(landmark_list, h, w):
    space = 25
    x_coord = []
    y_coord = []
    for landmark in landmark_list:
        x_coord.append(landmark.x)
        y_coord.append(landmark.y)

    return round(min(x_coord) * w) - space, round(min(y_coord) * h) - space, round(max(x_coord) * w) + space, round(max(y_coord) * h) + space

def predict_letter(image, model):
    image = image/255.0
    return categories[tf.argmax(model.predict(tf.expand_dims(cv2.resize(image, (400, 400)), axis= 0))[0]).numpy()]


def draw_hands(imgae):
    height, width = imgae.shape[:-1]

    hand_landmarks = detector.process(imgae)

    if hand_landmarks.multi_hand_landmarks:
        for num, hand in enumerate(hand_landmarks.multi_hand_landmarks):
            mp_draw.draw_landmarks(imgae, hand, mp_hands.HAND_CONNECTIONS)
            # xmin, ymin, xmax, ymax = calcBoundaryBox(hand.landmark, height, width)
            coord = calcBoundaryBox(hand.landmark, height, width)
            return cv2.rectangle(imgae, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 0), 4) , coord
            
    else:
        coord = None
        return imgae , coord
    
def draw_prediction(image, coord ,model):
    letter = predict_letter(image[coord[1] : coord[3], coord[0]:coord[2]], model)
    cv2.putText(image, letter, (coord[0] + 5, coord[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    
    return letter
