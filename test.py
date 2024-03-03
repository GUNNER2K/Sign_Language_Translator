import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

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

def hand_capture(img):
    detector = HandDetector(maxHands=1)
    safezone = 120
    hands, img2 = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgcrop = img[y-safezone:y+h+safezone, x-safezone:x+w+safezone]
        return tf.image.resize(imgcrop, (200, 200)) , True
    else:
        return img , False
    
from tensorflow.keras.models import load_model
import tensorflow as tf


def load_model():
    model = tf.keras.saving.load_model(
        'Models/asl_model_1.h5', custom_objects=None, compile=False, safe_mode=True
    )
    return model
model = load_model()    

cap = cv2.VideoCapture(0)
        
while cap.isOpened():
    ret, frame = cap.read()
    # print(frame)

    new_img , ishand = hand_capture(frame)
    if ishand:
        # print(new_img)
        # img = tf.image.resize(new_img , (200 , 200))
        # img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(new_img, axis=0)
        prediction = model.predict(img)
        prediction = np.argmax(prediction,axis=1)
        letter = categories[prediction[0]]
        print(letter)
        # prediction = model.predict(crop_img)
        ishand = False
    else :
        print("  ")
    cv2.imshow('frame' , frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()