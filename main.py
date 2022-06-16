import numpy as np
import HandTrackingModule as htm
import cv2
import mediapipe
import time
import autopy

wCam, hCam = 640, 480
wScr, hScr = autopy.screen.size()
frameR = 100 # Frame Reduction
smoothening = 7

previousLocX, previousLoxY = 0, 0
currentLocX, currentLocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
previousTime = 0
detector = htm.HandDetector(maxHands=1)

while True:

    # Find Hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Get tip of index and middle fingers.
    if len(lmList) != 0:
        # Coordinates of index finger
        x1, y1 = lmList[8][1:]
        # Coordinates of middle finger
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp(lmList)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # If only Index Finger is up (Moving Mode)
        if fingers[1] == 1 and fingers[2] == 0:
            # Converting Coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # Smoothening Values for moving mouse
            currentLocX = previousLocX + (x3-previousLocX) / smoothening
            currentLocY = previousLoxY + (y3-previousLoxY) / smoothening

            # (wScr-x3) is done to flip the mouse action for x-axis.
            autopy.mouse.move(wScr-currentLocX, currentLocY)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            previousLocX, previousLoxY = currentLocX, currentLocY

        # If both index and middle fingers are up (Clicking Mode)
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, _ = detector.findDistance(8, 12, img)
            # Click Mouse if distance is short
            if length < 40:
                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # Frame Rate
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(img, f"FPS-{str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 3)

    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)

