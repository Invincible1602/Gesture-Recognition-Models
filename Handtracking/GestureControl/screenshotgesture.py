import cv2 as cv
import numpy as np
import pyautogui
import time
import mediapipe as mp
import os
import handtrackingmodule as htm

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set webcam dimensions
wCam, hCam = 648, 488

# Initialize video capture
capture = cv.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

# Initialize hand detector from the custom handtrackingmodule
detector = htm.handDetector(detectCon=0.7)

# Function to recognize the specific gesture (open hand with all fingers extended)
def recognize_gesture(lmList):
    if len(lmList) == 21:  # Ensure correct number of landmarks for hand
        thumb_tip = lmList[4][2]
        index_tip = lmList[8][2]
        middle_tip = lmList[12][2]
        ring_tip = lmList[16][2]
        pinky_tip = lmList[20][2]

        thumb_mcp = lmList[2][2]
        index_dip = lmList[7][2]
        middle_dip = lmList[11][2]
        ring_dip = lmList[15][2]
        pinky_dip = lmList[19][2]

        # Check if all finger tips are above their corresponding joints
        if (thumb_tip < thumb_mcp and
            index_tip < index_dip and
            middle_tip < middle_dip and
            ring_tip < ring_dip and
            pinky_tip < pinky_dip):
            return True
    return False

# Flag and timer variables
gesture_detected = False
pause_duration = 10  # seconds
pause_start_time = 0

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        print("End of video or error reading frame.")
        break

    # Detect hands and draw landmarks on the frame
    frame = detector.findHands(frame)
    
    # Get landmark positions without drawing (for gesture recognition)
    lmList = detector.findPosition(frame, draw=False)

    # Check for specific gesture (open hand with all fingers extended)
    if recognize_gesture(lmList) and not gesture_detected:
        print("Gesture recognized! Taking screenshot...")
        
        # Set flag and start timer
        gesture_detected = True
        pause_start_time = time.time()
        
        # Display message on frame
        cv.putText(frame, "Pause for {} seconds".format(pause_duration), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Take screenshot using pyautogui (for demonstration)
        screenshot = pyautogui.screenshot()
        screenshot = cv.cvtColor(np.array(screenshot), cv.COLOR_RGB2BGR)
        cv.imwrite("screenshot.png", screenshot)

    # Pause for specified duration after first gesture recognition
    if gesture_detected:
        current_time = time.time()
        elapsed_time = current_time - pause_start_time
        if elapsed_time >= pause_duration:
            gesture_detected = False
            print("Ready to work.")

    # Display video feed with annotated landmarks
    cv.imshow('Video', frame)

    # Exit the loop when 'd' key is pressed
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# Release the video capture object and close all OpenCV windows
capture.release()
cv.destroyAllWindows()
