# for computer vision - handtracking
import cv2
import imutils
from imutils.video import VideoStream
import mediapipe as mp
# for AC API
import pyaccsharedmemory as pac
# others
import numpy as np
import os

### == SET UP == 

STEER_SENSITIVITY = 2
LEFT_THRESHOLD = -0.2
RIGHT_THRESHOLD = 0.2

ac = pac.ACCSharedMemory()
cap = VideoStream(src=0).start()
cap.set(3, 1280)
cap.set(4, 720)
mpDrawing = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()

### == NORMALIZED SLOPE VAL TO AC STEERING ==

def setSteer(val):
    ac.physics.steer = np.clip(val, -1, 1)


### SCREEN DISPLAY

def displayText(steerInput):
    steerText = "Left" if steerInput < LEFT_THRESHOLD else "Right" if steerInput > RIGHT_THRESHOLD else "Straight"
    # (img to write on, text, (position of text), font, font scale, color #white, thickness)
    cv2.putText(img, steerText, (360, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

### == GAME LOOP ==

while cap.isOpened():
    state, rawImg = cap.read()
    cv2.waitKey(1) # for refresh -> also frame rate control (feed fw 1ms)
    img = cv2.flip(rawImg, 1) # 0 for vertical flip, 1 for horizontal
    img.flags.writeable = False # prevent accidental modifications -> np runs faster
    detectedHands = hands.process(img) # returns hand landmarks (21 3D points for each hand)
    handLMs = detectedHands.multi_hand_landmarks
    if handLMs and len(handLMs) == 2: 
        left, right = handLMs[0].landmark, handLMs[1].landmark
        width, height = img.shape[1], img.shape[0]
        # each hand landmark has 21 points, index 11 = base knuckle of middle finger (chosen because relatively static)
        # * width and * height because landmark returned as [0, 1] normalized coords -> to obtain img coords
        leftMiddleFingerX, leftMiddleFingerY = left[11].x * width, left[11].y * height
        rightMiddleFingerX, rightMiddleFingerY = right[11].x * width, right[11].y * height
        slope = (rightMiddleFingerY - leftMiddleFingerY) / (rightMiddleFingerX - leftMiddleFingerX)
        normalizedSlope = np.clip(slope *  STEER_SENSITIVITY, -1, 1) # amplify for steering sensitivity (this is very cool!!) scaled to [-1, 1]
        setSteer(normalizedSlope)
        displayText(normalizedSlope)
    cv2.imshow("Image Feed", img)   
    # check if 'q' pressed -> exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # free hardware resource -> disconnect from camera
cv2.destroyAllWindows() # close all opencv gui windows opened -> close display windows

    

    