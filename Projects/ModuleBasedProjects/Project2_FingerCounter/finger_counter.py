import cv2 as cv 
import time 
import HandTrackingModule as htm 

###########
width , height = 640, 480
##########

cap = cv.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

pTime =0
detector = htm.HandDetector()

tipIds = [4,8,12,16,20]
while True : 
    success, img = cap.read() 
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    fingers = []

    if (len(lmList) != 0 ) : 
        #Thumb 
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1] : 
            fingers.append(1)
        else : 
            fingers.append(0)
        
        # 4 Fingers 
        for id in range(1,5) : 
            tipId = tipIds[id]
            if lmList[tipId][2]<lmList[tipId-2][2] : 
                fingers.append(1)
            else : 
                fingers.append(0)
        

        totalFingers = fingers.count(1)
        print(totalFingers)
        cv.putText(img, f"Fingers Count : {totalFingers}",(50,20),1,1,(255,0,0),2)

    cTime = time.time() 
    fps = 1/(cTime - pTime)
    pTime = cTime 

    cv.putText(img, f"FPS : {int(fps)}",(470,20),1,1,(255,0,0),2)

    cv.imshow("Image",img)
    cv.waitKey(1)