import numpy as np
import cv2
import imutils
from imutils.video import VideoStream
import time
import mediapipe as mp  # Ensure MediaPipe is imported

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Start the video stream (wait a bit for it to initialize)
print("Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)  # Allow camera to warm up

while True:
    img = vs.read()
    if img is None:
        print("Error: Failed to capture frame.")
        continue

    img = cv2.flip(img, 1)  # Flip for mirror effect
    img.flags.writeable = False  # Disable writing for processing optimization
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Enable writing back for drawing
    img.flags.writeable = True  

    # Image dimensions
    width, height = img.shape[1], img.shape[0]

    # If hands are detected
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        hand1 = results.multi_hand_landmarks[0].landmark
        hand2 = results.multi_hand_landmarks[1].landmark

        # Determine which hand is actually on the left
        if hand1[12].x < hand2[12].x:  # If hand1's middle finger tip is further left
            left_hand = hand1
            right_hand = hand2
        else:
            left_hand = hand2
            right_hand = hand1

        # Get middle finger tip positions
        left_mFingerX, left_mFingerY = int(left_hand[12].x * width), int(left_hand[12].y * height)
        right_mFingerX, right_mFingerY = int(right_hand[12].x * width), int(right_hand[12].y * height)

        # Draw circles on middle finger tips
        cv2.circle(img, (left_mFingerX, left_mFingerY), 10, (0, 255, 0), -1)
        cv2.circle(img, (right_mFingerX, right_mFingerY), 10, (0, 255, 0), -1)

        # Draw line connecting both fingers
        cv2.line(img, (left_mFingerX, left_mFingerY), (right_mFingerX, right_mFingerY), (0, 255, 255), 3)

        # Calculate delta values
        delta_x = right_mFingerX - left_mFingerX
        delta_y = right_mFingerY - left_mFingerY

        # Calculate angle in degrees using arctan2 and normalize to [-90,90]
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        # Normalize angle to a -1 to 1 range (1 corresponds to 90 degrees)
        normalized_angle = np.clip(angle / 90, -1, 1)

        # Determine steering direction using normalized angle
        if normalized_angle < -0.2:
            direction = "LEFT"
            color = (0, 0, 255)  # Red for left
        elif normalized_angle > 0.2:
            direction = "RIGHT"
            color = (255, 0, 0)  # Blue for right
        else:
            direction = "STRAIGHT"
            color = (0, 255, 0)  # Green for straight

        # Display angle value and direction
        cv2.putText(img, f"Angle: {angle:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Direction: {direction}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show camera feed
    cv2.imshow("Hand Slope Tracking Test", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
vs.stop()
cv2.destroyAllWindows()
