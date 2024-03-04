import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
import memory_profiler

tf.config.experimental.set_memory_growth(device= 'CPU', enable= True)

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

@memory_profiler.profile
def hand_capture(img):
    detector = HandDetector(maxHands=1)
    safezone = 120
    hands, img2 = detector.findHands(img)
    if hands:
        hand = hands[0]
        del hands, img2
        x,y,w,h = hand['bbox']
        imgcrop = img[y-safezone:y+h+safezone, x-safezone:x+w+safezone]
        # print(imgcrop.shape)
        if imgcrop.shape[0] != 0 and imgcrop.shape[1] != 0:
            return cv2.resize(imgcrop, (200, 200)), True, (x, y, w, h)
    return img, False, ()

@memory_profiler.profile
def draw_info_text(image, boundaries, letter):
    cv2.rectangle(image, (boundaries[0], boundaries[1]), (boundaries[0] + boundaries[2], boundaries[1] + boundaries[3]), (0, 0, 0), 1)

    letter_text = letter

    cv2.putText(image, letter_text, (boundaries[0] + 5, boundaries[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
    
    return image


@memory_profiler.profile
def load_model():
    model = tf.keras.saving.load_model(
        'Models/asl_model_1.h5', custom_objects=None, compile=False, safe_mode=True
    )
    return model

@memory_profiler.profile
def model_predict(img):
    # Preprocess img with TensorFlow functions outside the loop
    # img = tf.image.resize(img, (200, 200))
    # img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    prediction = np.argmax(prediction, axis=1)
    letter = categories[prediction[0]]
    del prediction, img
    return letter

model = load_model()    

cap = cv2.VideoCapture(0)    
count = 0
while cap.isOpened():
    count += 1
    ret, frame = cap.read()

    new_img , ishand, boundaries = hand_capture(frame)
    if ishand:
        letter = model_predict(new_img)
        time.sleep(0.02)
        draw_info_text(image= frame, 
                       boundaries= boundaries, 
                       letter= letter)
    else :
        print("  ")
    cv2.imshow('frame' , frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count == 3:
        break

del model
cap.release()
cv2.destroyAllWindows()