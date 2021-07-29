Here we will be using open-source mediapipe framework by Google to perform some easy tasks in computer vision like hand tracking, face detection, pose estimation in real time.  

#### Hand Landmarks 

![Hand Landmarks](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)


#--------------------------------------------------

#### Pose Landmarks 
![Pose Landmarks](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)

 

#### [mediapipe framework -- Google](https://mediapipe.dev/) 


##### Hand Tracking Module structure 
```
├── mediapipe
├   ├──solutions 
├        ├──hands
├            ├──Hands class
├                ├──process
├                    ├──multi_hand_landmarks
├            ├──HAND_CONNECTIONS
├        ├──drawing_utils
├            ├──draw_landmarks 
```

##### Pose Estimation Module structure

```
├── mediapipe
├   ├──solutions 
        ├── pose 
            ├── Pose class
                ├──process 
                    ├── pose_landmarks
            ├── POSE_CONNECTIONS 
    ├──drawing_utils
        ├──draw_landmarks
```

##### Face Detection Module structure 

```
├── mediapipe 
    ├── solutions 
        ├── face_detection 
            ├── FaceDetection class 
                ├── process 
                    ├── detections 
                        ├── score 
                        ├── location_data.relative_bounding_box    
```




