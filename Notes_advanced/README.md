### Advanced Computer Vision 

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



