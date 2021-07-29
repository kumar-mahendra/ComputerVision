import cv2 as cv 
import mediapipe as mp
import time 

class PoseEstimator : 
    def __init__(self, mode=False , upBody = False , smooth = True, detectionCon = 0.5 , trackCon = 0.5 ) : 
        self.mode = mode 
        self.upBody = upBody  
        self.smooth = smooth 
        self.detectionCon = detectionCon 
        self.trackCon  = trackCon 

        self.mpPose = mp.solutions.pose 
        self.pose  = self.mpPose.Pose(self.mode,self.upBody,self.smooth,self.detectionCon, self.trackCon) 
        self.mpDraw = mp.solutions.drawing_utils 
    
    def findPose(self, img, draw=True) : 
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return self.lmList
    

def main() : 
    cap = cv.VideoCapture(0)
    detector = PoseEstimator()

    pTime = 0 
    while True : 
        success, img = cap.read() 
        
        img = detector.findPose(img)
        lmList = detector.findPosition(img,draw=False)
        
        cTime = time.time() 
        fps = 1/(cTime-pTime)
        pTime = cTime 

        cv.putText(img, str(int(fps)), (10,50),0,1,(255,0,0),2)
        cv.imshow("Image",img)
        cv.waitKey(1)



if (__name__=='__main__')  : 
    main() 
