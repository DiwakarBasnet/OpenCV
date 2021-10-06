import cv2
import time
import PoseModuel as pm

cap = cv2.VideoCapture('PoseVids/1.mp4')
pTime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    if (cTime - pTime) != 0:
        fps = 1 / (cTime - pTime)
    else:
        fps = 0
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow('Image', img)

    cv2.waitKey(1)