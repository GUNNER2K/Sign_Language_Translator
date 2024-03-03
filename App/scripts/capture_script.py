import cv2
from cvzone.HandTrackingModule import HandDetector

def hand_capture(img):
    detector = HandDetector(maxHands=1)
    safezone = 30
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgcrop = img[y-safezone:y+h+safezone, x-safezone:x+w+safezone]
        return imgcrop, img