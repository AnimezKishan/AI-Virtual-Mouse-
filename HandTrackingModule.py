# The installation of following library is required to run this code in Python:
# pip install opencv-python
# pip install mediapipe

# if throws error after installation of mediapipe
# pip3 install --upgrade protobuf==3.20.0

import cv2
import mediapipe as mp
import time
import math

class HandDetector:
    def __init__(self, mode=False, maxHands=2, modComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modComplexity = modComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Setting up to specifically track hands.
        self.mpHands = mp.solutions.hands
        # Using Hands() from mediapipe module.
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modComplexity, self.detectionCon, self.trackCon)
        # For drawing tracking paths of hand
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        # Image converted to RGB Form.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # If Hand Landmark Detected
        if self.results.multi_hand_landmarks:
            # for every hand landmarks as handLms
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Tracks Hand if any hand is found on screen.
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True, handNo=0, target=8):
        self.lmList = []
        # Parameters for Boundary Box for Hand
        xList = []
        yList = []
        bbox = []
        if self.results.multi_hand_landmarks:
            # To Target mentioned hand.
            spHand = self.results.multi_hand_landmarks[handNo]
            # The following code is for targeting any specific landmark of hand
            for id, lm in enumerate(spHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])


            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            # To Draw Boundary Box.
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
        return self.lmList, bbox

    def fingersUp(self, lmListt):
        fingers = []
        # Thumb
        if lmListt[self.tipIds[0]][1] > lmListt[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if lmListt[self.tipIds[id]][2] < lmListt[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
def main():

    previousTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    # take 'tar' variable as input to target any specific landmark.
    ch = input("Want to target any specific landmark of hand? (y/n): ")
    if ch == 'y' or ch == 'Y':
        tar = int(input("Enter the landmark no. to be targeted: "))
    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        lmList, bbox = detector.findPosition(img, target=tar)
        if len(lmList) != 0:
            print(lmList[tar])

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()