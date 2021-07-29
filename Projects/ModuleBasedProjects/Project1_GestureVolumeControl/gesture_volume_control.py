import HandTrackingModule as htm 
import cv2 as cv
import numpy as np  
import time 
import math 

'''
PyCaw library is used to change volume based on distance between tip of index finger and tip of thumb 

https://github.com/AndreMiras/pycaw

pip install pycaw 

'''

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


#######
width, height = 640, 480
#######

cap = cv.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

detector = htm.HandDetector(detectionCon=0.75)


volRange = volume.GetVolumeRange()  # to get volume range 
print(volRange)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

# volume.SetMasterVolumeLevel(-20.0, None)  # to set volume to a specific value 


pTime = 0 
while True : 
    success, img = cap.read() 
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)

    if (len(lmList) != 0 ) : 

        x1, y1 = [lmList[4][1],lmList[4][2]]
        x2, y2 = [lmList[8][1],lmList[8][2]]
        cx, cy = (x1 + x2 )//2, (y1 + y2)//2 
        
        cv.circle(img,(x1,y1) ,5,(255,0,0),cv.FILLED)
        cv.circle(img, (x2,y2),5,(255,0,0),cv.FILLED)
        cv.line(img, (x1,y1),(x2,y2), (255,0,0),2 )
        cv.circle(img, (cx,cy),5,(255,0,0),cv.FILLED)

        length = math.hypot(x1-x2,y1-y2) 
        
        # my Hand range of lenght : 30 to 220 
        # volume range -65 to 0 

        # rescale 
        vol = np.interp(length, [30,220], [minVol, maxVol])
        volBar = np.interp(length, [30, 220], [400, 150])
        volPer = np.interp(length, [30, 220], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)  # change volume based on distance between fingers
        if length < 30:
            cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)

    # Add vertical bar showing current volume in % 
    cv.rectangle(img, (40,150),(70,400),(255,0,0),3)
    cv.rectangle(img, (40,int(volBar)),(70,400),(255,0,0),cv.FILLED)
    cv.putText(img, f"{int(volPer)}%",(50,450),1,2,(255,0,0),2)

    cTime = time.time() 
    fps = 1/(cTime - pTime)
    pTime = cTime 

    cv.putText(img, f"FPS : {int(fps)}",(20,50),1,2,(255,0,255),2)
    cv.imshow("imgae",img)
    cv.waitKey(1)

