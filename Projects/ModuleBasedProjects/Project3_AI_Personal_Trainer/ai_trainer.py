import cv2 as cv 
import time 
import PoseModule as pm 
import numpy as np 


cap = cv.VideoCapture(0)
pTime = 0 
detector = pm.PoseEstimator(False,False,True,0.75,0.75)
per = 0 
dir = 0 
count = 0 
while True : 
    success, img = cap.read() 
    img  = detector.findPose(img,False)
    lmList = detector.findPosition(img,draw=False)
    if (len(lmList) != 0) : 
        # # Left Arm 
        # angle = detector.findAngle(img, 11, 13, 15)

        # Right Arm
        angle = detector.findAngle(img, 12, 14, 16)

        per = np.interp(angle, (50,160), (100, 0))
        bar = np.interp(angle, (50,160), (100, 300))


        # Check for the dumbbell curls ( up + down)
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)

        # Draw Bar
        cv.rectangle(img, (550, 100), (600, 300), color, 3)
        cv.rectangle(img, (550, int(bar)), (600,300), color, cv.FILLED)
        cv.putText(img, f'{int(per)} %', (550,350), cv.FONT_HERSHEY_PLAIN, 4,
                    color, 4)

        # Draw Curl Count
        cv.rectangle(img, (10,300), (50, 400), (0, 255, 0), cv.FILLED)
        cv.putText(img, str(int(count)), (10, 380), cv.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 3)


    cTime = time.time() 
    fps = 1/(cTime - pTime)
    pTime = cTime 
    cv.putText(img, f"FPS : {int(fps)}",(20,50),1,2,(255,0,255),2)
    cv.imshow("image",img)
    cv.waitKey(1)
