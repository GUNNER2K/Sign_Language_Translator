from rembg import remove 
from PIL import Image 
import cv2 
import numpy as np
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = remove(frame)
    
    cv2.imshow('Hand Boundary Box', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()