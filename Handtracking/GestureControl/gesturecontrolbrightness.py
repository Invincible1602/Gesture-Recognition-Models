import cv2 as cv
import numpy as np
import time
import os
import handtrackingmodule as htm
import math
import screen_brightness_control as sbc

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set webcam dimensions
wCam, hCam = 648, 488

# Initialize video capture
capture = cv.VideoCapture(0)
capture.set(3, wCam)  # Set width
capture.set(4, hCam)  # Set height

pTime = 0  # Initialize previous time for FPS calculation
cTime = 0  # Initialize current time for FPS calculation

# Create an instance of handDetector with detection confidence of 0.7
detector = htm.handDetector(detectCon=0.7)

# Initialize brightness control
brightness = sbc.get_brightness(display=0)
minBright = 0  # Minimum brightness
maxBright = 100  # Maximum brightness
bright = 100  # Initial brightness value
brightBar = 400  # Initial position of the brightness bar
brightPer = 0  # Initial brightness percentage

while True:
    isTrue, frame = capture.read()  # Read a frame from the webcam
    if not isTrue:
        print("End of video or error reading frame.")
        break

    frame = detector.findHands(frame)  # Detect hands in the frame
    lmList = detector.findPosition(frame, draw=False)  # Get the positions of hand landmarks

    if len(lmList) != 0:
        # Get the positions of the thumb tip (id 4) and the index finger tip (id 8)
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Calculate the center point between thumb and index finger

        # Draw circles on the thumb tip and index finger tip
        cv.circle(frame, (x1, y1), 15, (255, 0, 255), cv.FILLED)
        cv.circle(frame, (x2, y2), 15, (255, 0, 255), cv.FILLED)
        # Draw a line between the thumb tip and index finger tip
        cv.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        # Draw a circle at the center point
        cv.circle(frame, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        # Calculate the length between the thumb and index finger
        length = math.hypot(x2 - x1, y2 - y1)

        # Convert hand range to brightness range
        bright = np.interp(length, [50, 230], [minBright, maxBright])
        brightBar = np.interp(length, [50, 300], [400, 150])
        brightPer = np.interp(length, [50, 300], [0, 100])

        sbc.set_brightness(bright)  # Set screen brightness

        # Change the center circle color based on the length
        if length < 50:
            cv.circle(frame, (cx, cy), 15, (0, 255, 0), cv.FILLED)
        elif length > 200:
            cv.circle(frame, (cx, cy), 15, (0, 0, 255), cv.FILLED)

    # Draw brightness bar
    cv.rectangle(frame, (50, 150), (85, 400), (0, 255, 255), 3)
    cv.rectangle(frame, (50, int(brightBar)), (85, 400), (0, 255, 255), cv.FILLED)
    cv.putText(frame, f'Bright: {int(brightPer)}%', (40, 450), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frame, f'FPS: {int(fps)}', (40, 70), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv.imshow('Video', frame)

    # Exit the loop when 'd' key is pressed
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# Release the video capture object and close all OpenCV windows
capture.release()
cv.destroyAllWindows()
