# Records

## Date

### Questions
- [ ] Add Superpoint repo in my repo.


## 4.02

### Sorting Project Pipeline 

#### Mapping

If we only consider one robot

There can be two kind of input data.
a. A sequence of images with order
b. A sequence of images without order 

a. If the input data is one sequence of images with order
Work Flow: 
1. Feature Extraction - Use superpoint to extract feature points of **each frame**
2. Feature Matching - Every two frame
3. Use matched information to generate initial estimation for poses
4. Triangulation to generate initial estimation for points
5. Use initial estimation for poses, initial estimation for points, and projection factor (poses and points correspondence) to build factor graph

#### Localization

Use the position of the previous pose.

#### Descriptors

With all the matched information and unmatched information, we can create a large feature descriptor visualization to observe how a group of descriptors act

#### Dataset

    1. matched information only of nearby frames
    2. matched information of every two frames

### Computational Photography Video Stabilization and Tracking Notes (This can be used for camera motion estimation.)

- gradient descent
- Gaussian Newton 
- matrix do second derivative 
- Foreground object and back ground objects, in my case background motion is the robot motion
    - decide motion model
        - translation 
        - similarity 2D
        - homography skew and perspective
        - similarity transform in 3D: x,y,z,r,p,y,s

- argmin sum(L2 norm) is average, while argmin sum abs is median 
    - median can discard outlier (like foreground dynamic object)
- `nuro` a company use gtsam to do selfdriving

- last kalman filter

- Understanding of video stabilization
    1. First use tracking to track the actual motion(camera intending motion and shake motion) from past frame to the current frame
        - To calculate the motion:
            1. calculate the motion vector of each 
            2. calculate the argmin sum(abs) of all motion vectors to get the background motion
            3. Delete all background motion from the motion vector to get the dynamic object motion
            4. Remove the dynamic object motion and will get all background object
            5. calculate the argmin sum(L2 norm) of all background object motion vectors to get the camera motion   
    2. Remove the motion of the current frame to get a static image.
    3. Do perspective transform to get a panorama with the past frame and the static image
    4. a static crop rectangle will get the actual next frame output 


### Questions

- [ ] What is the frame distance for Superpoint features to match
- [ ] Where do we need to add RANSAC in mapping to reduce error

### Process

- Hardware
There are two types of transistor:
One type is JFET(N-JFET & P-JFET) and one type is BJT (NPN & PNP) 
https://www.build-electronic-circuits.com/h-bridge/


## 4.3 Sort Inquiries

- [ ] Check and write code to make sure the feature matching information can be store and read
- [ ] Store descriptors into Map
- [ ] Will FOV equal scale transform when downsampling an image
- [ ] How should the north, west, east, south match with the x,y,z of the world coordinate of the map
- [ ] How to calculate the initial pose estimation?
    - [ ] Use matched features of the past pose and the current pose to get the R,t
- [ ] How to calculate the initial point estimation?
    - [ ] Average all seen landmark points, but in our case, we just do back projection
- [ ] Is it better to store feature and descriptors in list or map?
    - 
- If map input sequence is random then a tree structure is needed

- [ ] Create detail readme for dataset.

## 4.4 Create mutual pipeline for mapping and localization

### Crappy Mapping

- [ ] Collect data (a sequence of images with order)
    - images
    - ground truth poses
- [ ] Manually find the matched feature points
- [ ] Use Atrium_SfM to create map
- [ ] Use similarity transform to check the result

### Dataset for Varun

- [ ] Create cvs files to store match information
- [ ] Try to read cvs files

### Localization

- [ ] Collect data for localization
- [ ] test localization with the crappy map, only use **feature distance** for feature matching
- [ ] test localization with the crappy map, use **feature distance** and **descriptor L2 distance** for feature matching
- [ ] Use `dot product` instead of `L2 distance`
- [ ] Analysis `descriptor`

### Mapping Updata

- [ ] SfM front end
    - [ ] feature extraction for every frames
    - [ ] feature matching for every pair of frames
    - [ ] Pose initial estimation
    - [ ] Point initial estimation

### Localization Update

- [ ] Tracking - assume pose does not move a lot, track features to get camera motion 
- [ ] Multi threads
    - [ ] tracking for local localization 
    - [ ] Matching for global localization 
- [ ] Combine local localization with global localization through Kalman Filter

### Motion recover with Tracking

- Feature Extraction
- Track feature for 3 frames (all together 3 frames)
- Use the first frame and the last frame matched features to estimate an Euclidean transformation motion model with RANSAC, to calculate the camera motion

## 4.5

### mapping_front_end

#### Questions

- [ ] Can I get the correct feature if I down sample image with a scale for Superpoint, then up sample the output feature point with the same scale?
    - If process in this process, the regenerated keypoints are not totally the same as the key points generated without the down sampling process

- [ ] Array, list, or dictionary?
    - I perfer to use list to store keypoints and descriptors of one frame.
    - I perfer to use list to store feature information of different frames.
    - Array only need to be used when matching descriptors

- [ ] cv2.findFundamentalMat need to have at least a certain number of matches.

- [ ] The unit test for this part is hard to generate.

- [ ] There is a question when finding landmark correspondence.
    pose1  [1       ,2   ,3   ,4,None,None    ,None]
    pose2  [1       ,2   ,3   ,4,5   ,None    ,None]
    pose3  [`6 or 1`,2   ,None,4,5   ,`6 or 1`,None]
    pose4  [`6 or 1`,None,None,4,None,`6 or 1`,None]
    When pose3 and pose4

- [ ] There is gtsam.triangulatePoint3() in triangulation.h



### Tracking

# 4.6

## Questions

- [x] np.dot and matrix multiplication
    -Dot product of vectors and matrices (matrix multiplication) 

- [ ] How to know if whether a pair of matched features will generate a landmark that already exist or whether it is new?
    - [ ] If we do know how to judge this landmark, we can add a new landmark to the landmark list if this is a new landmark
    But, if the landmark already exist, how should we merge the landmark and all the matched information connected to the current matched features and the already matched features.

- [ ] Camera Pose can not be recovered if the camera only do rotation without translation? 
    - [ ] How to solve this problem？

- Homography matrix calculation problem. (The four pairs of corresponding points should be on the same surface.)
- How to know how many corresponding points needed?
- Easy way for tracking?

# 4.7 

## Progress

- [x] Get the ground truth of the features
- [x] Superpoint extract features
- [x] Manually match features 
- [-] Test SfM
    - Sim3 is wrong
    - features are bad

- [x] Save average descriptors then normalize the average and store it in a list
- [x] Create trajectory estimator input data
- [x] Test trajectory estimator front end 
- [-] Test Localization without descriptor matching 
- [x] Test Localization with descriptor matching

- [-] sim3 transform to the same coordinate as map
- [-] plot


# 4.9

## Progress

- [ ] Test SfM
    - Sim3 is wrong
    - features are bad

- [ ] Test Localization without descriptor matching 

- [x] sim3 transform to the same coordinate as map
- [x] plot



# 4.10

## Progress


- [ ] Check Sim3 is wrong
- [ ] Create multi pose pairs sim3 align function
- [ ] check if the matched features during the mapping example are bad
- [ ] unit test for Essential matrix, recover Pose(All opencv functions)
- [ ] Check the whole pipeline

- [ ] Test Localization without descriptor matching 

# 4.11

- [x] How does resizing an image affect the intrinsic camera matrix?
- https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix

## Progress

- [ ] Write unittest for opencv APIs
- [ ] Camera calibration
- [ ] Hardware 
https://www.alanzucconi.com/2015/08/19/how-to-hack-any-ir-remote-controller/?fbclid=IwAR187aUg2ciNrClwFw-Ca92bjID0h0S6kcmWqoZUciH5PJZwZ6AkY3SHJR4
- [ ] Sim3

# 4.12
- https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga13f7e34de8fa516a686a56af1196247f
- read nonlinear optimization and graph optimization，rewrite sim3, collect data
- prepare for localization
- camera lens distortion removal
- https://hackaday.io/project/12384-autofan-automated-control-of-air-flow/log/41862-correcting-for-lens-distortions

# 4.13
## Questions:
- Will undistort function change the camera intrinsic matrix?

## Progress:
- Add detail information for camera calibration

# 4.15
## Question
- [x] Downsample image will not change the distortion coefficient.
    - https://stackoverflow.com/questions/44888119/c-opencv-calibration-of-the-camera-with-different-resolution

# 4.16
## 
- [ ] Mapping
    - [ ] find essential
    - [ ] pose recover
    - [ ] triangulation
- [ ] Rematch data
- [ ] Solve triangulation
- [ ] Re-collect data

- [ ] What is the projection matrix created by the ros calibration node? 
    - [ ] the projection matrix is created byt the opencv getOptimalNewCameraMatrix() function: http://wiki.ros.org/image_pipeline/CameraInfo

- [ ] Localization
    - [ ] How is the projection working?
    - [ ] How is the distance working?
    - [ ] How is the feature descriptor working? 

- [ ] Important break through on Opencv. Please be careful of opencv camera coordinate.
- https://stackoverflow.com/questions/32889584/opencv-triangulatepoints-handedness
- https://ossyaritoori.hatenablog.com/entry/2017/06/25/Python_Opencv%E3%82%92%E7%94%A8%E3%81%84%E3%81%9F%E5%9F%BA%E6%9C%AC%E8%A1%8C%E5%88%97%EF%BC%8C%E5%9F%BA%E7%A4%8E%E8%A1%8C%E5%88%97%E3%81%AE%E6%8E%A8%E5%AE%9A

# 4.18
## Mapping
- `mapping_front_end_atrium` the basic for general sfm
- `mapping_a_line_with_self_corresponding` auto corresponding, input is an image sequence that face the same direction
- `mapping_with_self_corresponding` auto corresponding, input is an image sequence that token around a square 8*5*5

# 4.20
- SFM
https://github.com/adnappp/Sfm-python
- video stabilization
- https://www.learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/

- [x] Get average normalized descriptors
- [ ] Plot descriptors
- [x] Add transform to atrium map function
- [ ] Collect data for mapping and data for localization

# 4.22
- [x] Opencv sift problem solve:
https://stackoverflow.com/questions/52305578/sift-cv2-xfeatures2d-sift-create-not-working-even-though-have-contrib-instal


# 4.24
- [x] Work on test camera calibration
    - [x] is the `error` of the gtsam graph the reprojection error
        - No
    - [x] how to calibrate undistort images
        - Matlab toolbox

# 4.26
- [ ] test on my own code
- [x] Calibrate distort and undistort images
- [ ] Get the filter data for Varun
- [ ] Find ideas for localization
- [ ] Work on hardwares 


- [ ] understanding of 3D pose and transform
- [x] compare which calibration tool is the best and what is projection matrix?
    - if calibration is done by matlab then use matlab to undistort image
    - if calibration is done by ROS then use python to undistort image 
- [ ] understand how prior work

# 4.29

- [x] work on reprojection error on Runcam
    - reprojection does not work
    - wrote my own function to calculate reprojection 
    - question: There are still distortions after I calibrate the camera, should I continue to use Cal3_S2 or Cal3DS2
- [ ] work on reprojection error on Camera 2
- [x] contacted Varun
    - [ ] no reply
- [ ] create a script to automatically find correspondences for all images
- [ ] plot descriptor and analysis
