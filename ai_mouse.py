import cv2
import HandTrackingModule as htm
import numpy as np
import time
import autopy as ap

############
wCam = 640
hCam = 480
frameR = 100  # frame reduction
smoothing = 7

prevX, prevY = 0, 0
currX, currY = 0, 0

wScrn, hScrn = ap.screen.size()
########

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

previousTime = 0
currentTime = 0
detector = htm.HandTracker(maxHands=1)

while True:
    success, img = cap.read()
    img = detector.FindHands(img)
    lmlist, bbox = detector.FindPosition(img)

    # Find the tip of index and middle finger
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]  # index finger
        x2, y2 = lmlist[12][1:]  # middle finger

        # check which finger is up
        fingers = detector.FingersUp()
        # print(fingers)

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        ## index finger is up middle is down

        if fingers[1] == 1 and fingers[2] == 0:
            ## convert co-ordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScrn))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScrn))

            # smoothing the values to stop cursor flickering
            currX = prevX + (x3 - prevX) / smoothing
            curry = prevY + (y3 - prevY) / smoothing
            # move mouse
            ap.mouse.move(wScrn - currX, currY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            prevX, prevY = currX, currY
        ## both the index and middle finger are up
        if fingers[1] == 1 and fingers[2] == 1:

            ## check distance b/w two fingers
            length, img, lineInfo = detector.FindDistance(8, 12, img)

            ## clicking the mouse
            if length < 30:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
                ap.mouse.click()

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, f'FPS : {int(fps)}', (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (225, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
