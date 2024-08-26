import cv2 as cv
import numpy as np
import time
import os
import handtrackingmodule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
import tensorflow as tf

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set webcam dimensions
wCam, hCam = 648, 488

# Initialize video capture
capture = cv.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

pTime = 0
cTime = 0

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and enabled")
else:
    print("GPU is not available, using CPU")

# Create an instance of handDetector
detector = htm.handDetector(detectCon=0.7)  # for smoothness in detection

# Initialize audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()

vol = 0
volBar = 400
volPer = 0
minVol = volRange[0]
maxVol = volRange[1]

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        print("End of video or error reading frame.")
        break

    frame = detector.findHands(frame)  # Detect hands
    lmList = detector.findPosition(frame, draw=False)  # Get landmark positions

    if len(lmList) != 0:
        # Get the positions of the thumb tip (id 4) and the index finger tip (id 8)
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv.circle(frame, (x1, y1), 15, (255, 0, 255), cv.FILLED)
        cv.circle(frame, (x2, y2), 15, (255, 0, 255), cv.FILLED)
        cv.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv.circle(frame, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        # Calculate the length between the thumb and index finger
        length = math.hypot(x2 - x1, y2 - y1)

        # Convert hand range to volume range
        vol = np.interp(length, [50, 230], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])

        volume.SetMasterVolumeLevel(vol, None)

        # Change color based on the length
        if length < 50:
            cv.circle(frame, (cx, cy), 15, (0, 255, 0), cv.FILLED)
        elif length > 200:
            cv.circle(frame, (cx, cy), 15, (0, 0, 255), cv.FILLED)

    # Draw volume bar
    cv.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)
    cv.putText(frame, f'Vol: {int(volPer)}%', (40, 450), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

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
