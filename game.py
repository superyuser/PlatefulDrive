import cv2

import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    success, img = cam.read()
    img = cv2.flip(img, 1)
    img.flags.writeable = False # Making the images not writeable for optimization.
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Processing video.
    # Note: Converting to RGB results in a significant increase in hand recognition accuracy.
    cv2.imshow("Hand Recognition", img)

    # Checking if a hand exists in the frame.
    landmarks = results.multi_hand_landmarks # Fetches all the landmarks (points) on the hand.
    if landmarks:
        # When a hand exists in the frame.

        if(len(landmarks) == 2): # If 2 hands are in view.
            left_hand_landmarks = landmarks[1].landmark
            right_hand_landmarks = landmarks[0].landmark

            # Fetching the height and width of the camera view.
            shape = img.shape
            width = shape[1]
            height = shape[0]

            # Isolating the tip of middle fingers from both hands, and normalizing their coordinates based on height/width
            # of the camera view.
            left_mFingerX, left_mFingerY = (left_hand_landmarks[11].x * width), (left_hand_landmarks[11].y * height)
            right_mFingerX, right_mFingerY = (right_hand_landmarks[11].x * width), (right_hand_landmarks[11].y * height)

            # Calculating slope between middle fingers of both hands (we use this to determine whether we're turning
            # left or right.
            slope = ((right_mFingerY - left_mFingerY)/(right_mFingerX-left_mFingerX))

            # Outputs for testing.
#            print(f"Left hand: ({left_mFingerX}, {left_mFingerY})")
#            print(f"Right hand: ({right_mFingerX}, {right_mFingerY})")
#            print(f"Slope: {slope}")

            sensitivity = 0.3 # Adjusts sentivity for turing; the higher this is, the more you have to turn your hands.
            if abs(slope) > sensitivity:
                if slope < 0:
                    # When the slope is negative, we turn left.
                    print("Turn left.")
                if slope > 0:
                    # When the slope is positive, we turn right.
                    print("Turn right.")
            if abs(slope) < sensitivity:
                # When our hands are straight, we stay still (and also throttle).
                print("Keeping straight.")
            # Iterating through landmarks (i.e., co-ordinates for finger joints) and drawing the connections.
#        for hand_landmarks in landmarks:
#            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()