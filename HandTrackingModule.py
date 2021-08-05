import cv2

import mediapipe as mp

import time

import math

class HandTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon,
                                        self.trackCon)  # this class only uses RGB image
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def FindHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms,
                                               self.mpHands.HAND_CONNECTIONS)  # handlms in among multiple hands # here hand connection function is used to join the points
        return img

    def FindPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        xList = []
        yList = []
        bbox = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape  # height, width, channel of the image
                cx, cy = int(lm.x * w), int(lm.y * h)  # cx and cy position of center
                # print(id, cx, cy)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                # if id == 12:     #these are id id numbers among all 21 ids
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax

            if draw:
                    cv2.rectangle(img, (bbox[0]-20,bbox[1]-20),  (bbox[2]+20,bbox[3]+20),(255, 0, 255), 3)

        return self.lmList, bbox

    def FindDistance(self,p1,p2,img,draw = True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def FingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # totalFingers = fingers.count(1)

        return fingers


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandTracker()
    while True:

        success, img = cap.read()
        img = detector.FindHands(img)  # here you can put draw = False to stop plottiong the connections
        lmList = detector.FindPosition(img, )  # here you can put draw = False to stop plottiong the points
        # if len(lmList) != 0:
            #print(lmList[4])  # 4 is tip of thumb you can take value b/w 0 to 20 i.e 21 points on our palm
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS : {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (225, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
