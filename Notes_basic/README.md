
### Getting Started with OpenCV

<details>   
<summary> <b>Contents</b> 🔥</summary>
  <details>
  <summary>Introduction</summary>

  - [Description](#description)
  - [Packages](#packages)
  - [Import Modules](#import-modules)
  - [Basic Operations](#basic-operations)
  </details>
  <details>
  <summary>Image Processing Part 1</summary>

  - [Orientation convention](#orientation-convention)
  - [Resize](#resize)
  - [Cropping](#cropping)
  - [Blurring](#blurring)
  - [Color Spaces](#color-spaces)
  - [Edge Detection](#edge-detection)
  - [Morphological Operations](#morphological-operations)
  </details>
  <details>
  <summary>Image Processing Part 2</summary>
  
  - [Draw Shapes](#draw-shapes)
  - [putText](#putText)
  - [Bitwise Operations](#bitwise-operations)
  - [Transformations](#transformations)
  - [TrackBars](#trackbar)
  - [Contours](#contours)
  - [Masking](#masking)
  </details>

  <details>
  <summary>Resources</summary>

  - [Learning Resources](#learning-resources)
  </details>

</details>

#### Description
OpenCV is a computer vision library available in multiple programming languages including **python**.
computer vision help us to derive insights from media files i.e. images/videos.

#### Packages
- OpenCV-python version 3.7 or above  
- NumPy library 

#### Import Modules
```
import cv2 as cv      # opencv module 
import numpy as np    # Numpy module
```

#### Basic Operations 

**Read, Write & Show Images**
```
frame = cv.imread(<Path-to-Image>)  # to read image 

cv.imshow(frame)  # display image 

bool_Success = cv.imwrite(<Image-file-name-with-format>,frame)  # to save a image frame in current directory 
```
**Read, Write & Show Videos**
```
cap = cv.VideoCapture(<Cam-Index> or <Path-to-Video>)  # create object for video capture , use Cam-Index = 0 for using your laptop camera
success, frame = cap.read()  # read current frame of camera/video stream . Generally while loop is used to read frames continuously one at a time

# to display videos we need to show each frame individually like in  images
```
To save video file we need 5 things  
1. `Output-File-Name-With-format`
2. **[`FourCC-Code`](https://docs.opencv.org/4.5.2/dd/d43/tutorial_py_video_display.html)**
3. `isColor Flag` default: `True` image is expected to be colored
4. `fps` Frames Per second  
5. `size` of each frame we need as a tuple (frameWidth,frameHeight) 
```
four_cc = cv.VideoWriter_fourcc(<Four-CC Code>) 
video_writer = cv.VideoWriter(<Output-file-name-with-format>,four_cc,<fps>,<frameSize>,[,isColor])  # isColor is optional parameter 

```

*Note - Every time you exit the program make sure you release the camera (if used) and destroy all windows opened using these commands*

```
cv.release()
cv.destroyAllWindows()
``` 

##### Orientation Convention  
In OpenCV you should always keep in mind this thing
- -Ve Y-axis on coordinate axis is considered as positive Y-axis in OpenCV function . No change in X-axis convention . i.e. w.r.t. (0,0) our image lies in 4th quadrant . i.e height is measured from origin toward negative y-axis in coordinate geometry and width is measured from (0,0) towards positive x-axis as expected. 
- color images follows BGR convention . 

##### Resize 

We can do both shrink and stretch images. There are two ways to resize 
1. absolute way 
```
resizedImg = cv.resize(srcImg, (newWidth, newHeigth)) 
```
2. relative to (0,0) 
```
resizedImg = cv.resize(srcImg, (0,0), None, <scale-along-X-axis>,<scale-along-Y-axis> )
```

##### Cropping 
To perform cropping just perform slicing operation as we do in case of numpy array as image matrix is nothing but a numpy array 

To learn about slicing & indexing visit official docs [here](https://numpy.org/doc/stable/reference/arrays.indexing.html) 

##### Blurring  
Blurring/Smoothing is done by convoluting some spefic-type of filters(kernel) with image . [low-pass filters](https://en.wikipedia.org/wiki/Low-pass_filter)  are used for blurring . 

1. `GaussianBlur` - Use gaussian distribution of nearby pixels for filtering. nearer the point more is weight of pixel .

```
blurImg = cv.GaussianBlur(srcImg, ( <kernel_width>, <kernel_height> ), sigmaX [, sigmaY])  # if sigmaY not provided it take same value as sigmaX  .
``` 
*Note - In any kernel width and height should be positive and odd*

2. `Averaging`  
```
blurImg = cv.blur(srcImg, (<kernel_width>,<kernel_height>))
```

3. `Median`
```
blurImg = cv.medianBlur(srcImg, <kernel_size>) # kernel_size will be used for both width and heigth
```

4. `bilateral Filtering` - It is highly effective in removing noise while keeping edges sharp. it consider both gaussianDistribution as well as instensity of nearby pixels while performing filtering. 
```
blurImg = cv.bilateralFiltering(srcImg, <Filter-Size>, <sigma-Color>, <sigma-Space> ) # sigma-Color is for intensity range we want to consider  
```
For complete info about bilateralFiltering - [here](http://people.csail.mit.edu/sparis/bf_course/)


##### Color Spaces 
Switch from one color space to other . Default color space is BGR . 
```
newImg = cv.cvtColor(srcImg,<color-Code> )  # color codes examples cv.COLOR_BGR2GRAY , cv.COLOR_BGR2HSV, cv.COLOR_BGR2RGB etc. 
```

##### Edge Detection
1. `Canny edge dection` - Most popular edge dectection algo. This is a 4 stage process explained in simple language [here](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html) . 
noise_cancellation --> gradientsComputation --> non-max_supression --> Thresholding
```
cannyImg = cv.Canny(srcImg, <minThreshold>,<maxThreshold>)
```

2. `Simple Thresholding` - pixels with value smaller than threshold set to 0 and larger than threshold are set to maxVal 
```
thresImg = cv.threshold(srcImg, <threshold>,<maxVal>,thresholding_Type)  # options cv.THRESH_BINARY, cv.THRESH_TOZERO and their inverse , cv.THRESH_TRUNC, 
```
3. `Adaptive Thresholding` - based on intensity of colors automatically pick threshold values for different location using adaptiveMethod provided 
```
adapThresImg = cv.adaptiveThreshold(srcImg,<maxVal>,adaptiveMethod) # methods available are cv.ADAPTIVE_THRESH_MEAN_C, cv.ADAPTIVE_THRESH_GAUSSIAN_C
```

4. `Sobel & Scharr Derivates` - Gaussian Smooting + derivates used for noise reduction  
```
sobelImg = cv.Sobel(srcImg,cv.CV_64F,<bool_X>,<bool_Y>,<kernel-size>)
```

5. `Laplacian Derivatives` - Gaussian + Laplacian derivative used for noise reduction 
```
laplacianImg = cv.Laplacian(srcImg, cv.CV_64F)
```

##### Morphological Operations 

Those operations which process images based on shape/form of image . These operations apply some structuring operation on  image to get an output image . 

1. `Dilation` -   overlap Kernel and image and replace value of anchor point (usually center of kernel ) with maximum intensity in that region 
```
dilatedImg = cv.dilate(srcImg, ( <kernel_width> ,<kernel_height> ),iterations = <count> )
```
2. `Erosion` - complementry of dilation 
```
erodedImg = cv.erode(srcImg, ( <kernel_width> ,<kernel_height> ),iterations = <count> )
```

3. `Opening` - It is erosion followed by dilation . 
```
openImg = cv.morphologyEx(srcImg, cv.MORPH_OPEN, (<kernel_width>,<kernel_height>) )
```

4. `Closing` - It is reverse of Opening . 
```
closeImg = cv.morphologyEx(srcImg, cv.MORPH_CLOSE, (<kernel_width>,<kernel_height>) )
```

5. `Gradient` - difference between dilation and erosion of image 
```
gradientImg = cv.morphologyEx(srcImg, cv.MORPH_GRADIENT, (<kernel_width>,<kernel_height>) )
```
similarly there is `Top Hat`/`Black Hat` for difference between input image and Opening/Closing image.


##### Draw Shapes

```
circle = cv.circle(srcImg, <center_coordinate>, <radius>, <colorBGR>,<thickness>)

rectangle = cv.rectangle(srcImg, <top_left_diagonal_coors>,<bottom_right_diag_coors>, <colorBGR>, <thickness>)

line = cv.line(srcImg, <left_end_coors>, <right_end_coors>, <color>, <thickness>)
``` 

##### putText on Images
```
textImg = cv.putText(srcImg, <Text>, <origin_coors>, <fontFace>, <fontScale>, <colorBGR>, <thickness> )
```

###### fontFace - int or write explicitely font type . check available options [here](https://docs.opencv.org/4.5.2/d6/d6e/group__imgproc__draw.html#ga0f9314ea6e35f99bb23f29567fc16e11)
###### fontScale -  double,  scaling factor 


##### Transformations
Perform various transformations on 2D images.

1. `warpAffine`  - It uses a 2x3 transformation matrix, M for transformation.
     - `Translation` : M = [ [1,0,tx],[0,1,ty] ] tx, ty are translation along x and y axis resp. 
```
transformedImg = cv.warpAffine(srcImg, M, (<widthOut>,<heightOut>) )
```

2. `Rotation` - we also have a 2x3 transformation matrix for rotation. using openCV we can get that using cv.getRotationMatrix2D() method 
```
rot_M = cv.getRotationMatrix2D( <point_of_rotation>, <angle_of_rot>, <scalingFactor>)  
rotatedImg = cv.warpAffine(srcImg, rot_M, (<widthOut>,<heightOut>) ) 
```   

3. `Affine Transformation`  - Those transformation in which all parallel lines in original image remain parallel even after transformation 

```
M = cv.getAffineTransform(pts1,pts2)
affineImg = cv.warpAffine(srcImg, M, (<widthOut>,<heightOut>) )
```

4. `Perspective Transformation` - need a 3x3 transformation matrix. straight lines will remain straight . to find transformation matrix we need 4 pts on input image and **cooresponding** points on output image. points should be such that 3 of them should NOT be collinear. 
```
M  = cv.getPerspectiveTransform(pts1,pts2)
perspectiveImg = cv.warpPerspective(srcImg, M, (<widthOut>,<heightOut>) )
```

##### Bitwise Operations 
```
# Both images should be of same shape 
andImg = cv.bitwise_and(img1, img2) 
orImg = cv.bitwise_or(img1,img2)
xorImg = cv.bitwise_xor(img1, img2)
notImg = cv.bitwise_not(srcImg)
```

##### TrackBars 

```
cv.createWindow("Track_WindowName") #create window to see trackbars 

cv.createTrackbar("TrackBar_Name","Track_WindowName",<initial_val>, <maxVal>,someFunction)  # create a trackbar 

cur_val = cv.getTrackbarsPos("TrackBar_Name","Track_WindowName")  # get current value of trackbar 
```
##### Contours 
A Contour is basically a curve along the boundary joining continuous points having same intensity or color . It is preferred to use a binary image.  

###### Find
```
contours, hierarchy = cv.findContours(srcImg, <contour-retrievel-mode>, <contour-approximation-method>)  # find all contours with hierarchy 
```
###### Draw
```
cv.drawContours(outImg, <list-of-contours>, <index-list or -1>,<color>, <thickness> ) # -1 denotes show all contour present in list 
```

###### Features 

1. **[`Image Moments`](https://en.wikipedia.org/wiki/Image_moment)**  
moments are helpful in describing contour objects after segmentation. to know more visite official [docs](https://docs.opencv.org/4.4.0/d8/d23/classcv_1_1Moments.html).
``` 
M = cv.moments(cntour)  # dictionary of moments 
```

2. `Contour Area and Perimenter` 
```
area = M['m00] OR cv.contourArea(contour)
perim = cv.arcLength(contour,<bool_isClosed>)
```
3. `Contour approximation` 
```
approxPolygon = cv.approxPolyDP(curve, <epsilon>, <bool_isClosed>) # epsilon denote max distance between approxmated polygon and original curve
```
4. `Convex Hull` 
```
hull = cv.convexHull(contour)  # return coordinates of convex hull 
```
5. `Bounding Rectangle` 
 - Straight BR 
    ```
    x,y,w,h = cv.boundingRect(contour) # find top-left coordinate and width and height
    cv.rectangle(srcImg, (x,y),(x+w,y+h), <color>,<thickness>)
    ```

 - `Rotated BR 
    ```
    rotRect = cv.minAreaRect(cntour)
    box = np.int0(cv.boxPoints(rotRect))
    cv.drawContours(srcImg,[box],0,<color>,<thickness>)
    ```

6. `Hierarchy` - Parent child relationship(s) in contours 
in hierarchy it is list of arrays `[next, previous, firstChild, parent]`. If there is no child or parent it is taken as -1. 
It is decided by *retrivel-method* parameter in findContours() method of OpenCV.  
    - `RETR_LIST`  - all contours are at same levels 
    - `RETR_EXTERNEL` - only returns outermost level members 
    - `RETR_CCOMP` - make 2-level hierarchy . externel contours in hierarchy-1 and internel contours in hierarchy-2 
    - `RETR_TREE` -  It retrieves all the contours and creates a full family hierarchy list. It even tells, who is the grandpa, father, son, grandson and even beyond... :) Official docs calls it Mr. Perfect which it is indeed!!!

##### Masking 
```
maskImg = cv.inRange(imgHSV, lower, upper)  # create mask 
# lower --> [hue_min, saturation_min, value_min]
# upper --> [hue_max, saturation_max, value_max]

newImg = cv.bitwise_and(srcImg, srcImg, mask = maskImg)
```
#### Learning Resources 

1. **[Computer Vision Zone](https://www.computervision.zone/)**
2. **[Free Code Camp Computer Vision Course](https://youtu.be/oXlwWbU8l2o)** [`Git_Repo`](https://github.com/jasmcaus/opencv-course)
3. **[OpenCV Tutorials](https://docs.opencv.org/3.4/d6/d00/tutorial_py_root.html)**
4. **[Beginner Level Projects](https://www.computervision.zone/projects/)**



