# Records

## Date

### Questions

### Progress

- [ ] SFM feature point association (Using program instead of manual)
- [ ] trajectory estimator velocity and rotation accelerations

## 2.27

### Questions

- [ ] Will FOV change when downsampling images?

- [x] How should I calculate the area that landmarks can be captured

- [x] Because the output of Superpoint descriptors are arrays should I also use array to store feature points (2D list or 2D np.array)

- [ ] When doing functional programming and I want to create a trajectory that also preserve the previous poses, in this case should I create a function that use old trajectory and a new pose as input, and the ouput should be a new trajectory?
~~~
def trajectory_generator():
	input:
		- past_trajectory
		- new_pose

	output:
		- new_trajectory
~~~ 

### Progresses

- [-] data association unittest
- [-] trajectory_estimator unittest
- [x] study functional programming: http://www.lihaoyi.com/post/WhatsFunctionalProgrammingAllAbout.html
- [-] use submodule to substitute superpoint


## 2.28 & 3.1 & 3.2 

### Questions

- [ ] factor graph constraint
- [ ] Data association, is it better to have two side match or should I use one side match
- [x] How to save numpy array with commas, save in .txt document do not have commas.
~~~
a.tofile('foo.csv',sep=',',format='%10.5f')
# Or
np.savetxt('foo.csv',a,fmt='%10.5f',delimiter=',')
~~~
- [x] What are the ouput formats of superpoints and descriptors?
	- superpoints 3*N np.array 
	- descriptors 256*N np.array 

- [ ] As for SfM, how will the pose of the camera influence the result of the map.
	-  Changes of variables: x, y, z, row, pitch, yaw. Also, what is the difference between small changes and huge changes?
- [ ] (SfM) Can prior be calculated by using essential matrix to calculate R and T 

### Progress

- [x] Generate trajectory estimator unittest data.
- [-] Features and map compare, assert equal function
- [-] data association unittest
- [-] collect new data for SfM
- [-] trajectory_estimator unittest
- [-] use git submodule to substitute superpoint folder
- [-] Use display function to plot output of SfM


## 3.4

### Questions

- [x] How descriptors behave between different features, and how descriptors behave between the same features from different frames, closet frames and frames with large interval
	- same descriptors of two continuous frames features
		small L2 distance
	- same descriptors of the same features from two frames of two camera poses
		the further the poses are the larger the distance is
	~~~
	frame 2 and frame 3 are two continuous frames
	frame 9 is around 2 meters away from frame 3  
	frame 10 is around 2 meters away from frame 9  

	results:       
	2.1 - 3.1 : 0.0754817355391
	2.1 - 9.1 : 0.889656016896
	2.1 - 10.1 : 1.1497769248

	2.2 - 3.2 : 0.161638398285
	2.2 - 9.2 : 0.746780589062
	2.2 - 10.2 : 1.14186204753
	~~~
	- different descriptors within the same frame
		L2 distances  most `>0.9`, some `[0.7,0.9]` some `[0.6, 0.7]` and 1 `0.49`
- [ ] Is the SfM from GTSAM so sensitive?

- [ ] How to store data list, array, or GTSAM point2, point3
- [ ] What kind of data association problem will appear?
	* the map includes lots of landmark points will similar descriptors
	* two superpoint feature match to one landmark point, several superpoint features have similar descriptors
	* no superpoint feature
	* no matches

- [ ] I am actually going to use the previous pose projected features to match with my current frame superpoint features 

- [ ] If the trajectory estimator fail to generate the current pose, and the robot is keep moving then the difference past pose and the current pose will become larger and it will be harder to generate the current pose. 

- [ ] The project feature point coordinates and the superpoint coordinate will have large differences when the past pose and the current pose have a differences of `0.1` (delta in the create past pose function.)


### Progress

- [x] Create new normalized vector dataset: such as [0,0,1,0,0] [0,1,0,0,0]
- [x] Complete data association.
- [ ] `Features` and `Map` compare, assert equal function
- [ ] `Features` and `Map` append function
- [x] data association unittest


## 3.5

### Record

- calibration matrix can be thought as similarity transform : cP = cRw * wP
	* `cRw`: [[F,0,Ux],[0,F,Vy],[0,0,1]]

### Progress

- [-] unittest for group action similarity transform on poses coordinates (check sim3 in GTSAM
	- Robert Mahony Geometic for SLAM.pdf
- [ ] use constraint factors for trajectory estimator factor graph
- [ ] change the input of trajectory estimator from a pose to a trajectory
- [ ] tuple input


## 3.8

### Questions

- [ ] Differences between WiFi module as client and server?

- [ ] ESP8266 wifi module control 
	- https://www.youtube.com/watch?v=2AL7HfiRlp4&vl=en
	- https://tttapa.github.io/ESP8266/Chap07%20-%20Wi-Fi%20Connections.html
	- 	(Not important: https://www.youtube.com/watch?v=QVcpzwY4hWI)

### Control Progress

- [x] Power ESP8266 with battery 
- [x] Ping laptop with ESP8266 as a WIFI client through personal hotspot
- [x] Create a WiFi server with the ESP8266, connect both laptop and ESP8266 with personal hotspot.
	- Result display in chrome
	- https://www.youtube.com/watch?v=m2fEXhl70OY

- [x] ESP8266 GPIO 16 write voltage, and use positive voltage to control motor
	- https://www.youtube.com/watch?v=CpWhlJXKuDg

- [ ] ESP8266 GPIO index and the corresponding port

- [ ] ESP8266, how to generate negative voltage?


## 3.9

### Questions

- [x] Does `gtsam` has constraint factor?
	* Yes, it does but it is not wrapped by cython.
- [x] How to know which libraries are convert to python version?
	- The problem is sim3 seems to be not able to import. 
	* In `gtsam`->`cython`->`tests`->`gtsam_test.h`
- [x] Why are there two type of similarity transform structure?
	* One is [sR t; 0 1], other one is [R t; 0 1/s]
	* They are actually the same. Just the translation matrices are different. t(from the left matrix) = st(from the right matrix)

### Progress

- [ ] unittest for group action similarity transform on poses coordinates (check sim3 in GTSAM
	- Robert Mahony Geometic for SLAM.pdf

- [ ] Import Sim3
- [ ] similarity transform on Pose3 
- [ ] similarity transform calculation with `poses`
- [ ] constraint factor


## 3.10

### Questions

- [ ] Does gtsam->python->handwritten include all the classes that are wrapped into python in gtsam? 

### Progress

- [ ] Create map object to include both landmarks and the trajectory
- [ ] Create similarity3 class to generate similarity transform matrix will poses, and use similarity transform to calculate the transform map
- [ ] Create unittests
- [ ] Test the similarity transform class on the SFM output
- [ ] Modify trajectory estimator, `change the pose input into trajectory input` and `Document` the code.
- [ ] Refactor code and make a PR 

- [ ] Upload the panorama assignment to gradescope
- [ ] Finish English assignments
- [ ] Finish the Gas Association Document

## 3.11

### Questions

- [ ] Is the rotation matrix of poses also orthogonal?
- [ ] negative cubic root
- [ ] zero determinant discussion
- [ ] What if scalar is zero? scalars of homogeneous and transformation matrix
- [x] Finish similarity class function and unittests


## 3.24

### Progress
- [x] Finish sim2 align with point2 pairs
- [x] Finish sim3 align with pose3 pairs


## 3.25

### Questions

### Progress
- [-] Solve sim3 s == 0
- [x] sim3 map_transform()

## 3.27

### Questions

- [-] How to use sim3 in mapping
- [-] In localization, does the descriptor idea works?
- [-] CUrrent with the code we can only generate 3 states. How to generate 12 states? 
- [x] Group meeting for the Blimps.

### Progress

- [-] Finish the current branch
- [-] Solve and improve sim3 s == 0 
- [-] Add GTSAM unittest
- [-] Make a PR for Frank

## 3.28

### Progress

- [x] Solve python external module import problem
http://python-notes.curiousefficiency.org/en/latest/python_concepts/import_traps.html
https://blog.csdn.net/chinesepython/article/details/82113575
https://blog.csdn.net/weixin_38256474/article/details/81228492
- `Note`: The problem is caused by the difference between absolute path and relative path
	- If code is ran in a terminal, it is relative path, the root path starts at the current path and need to add `../`
	- If code is ran in VScode, it is also relative path but the root path starts at the `Blimps` 
- This will cause problems when importing or reading files or images


- [-] Refactor 
	- [x] atrium_sfm
		- atrium_sfm & test_atrium_sfm: refactor code and solve import problem.
		- sfm_data: Code refactor



## 3.29 & 3.30
### Question

- [ ] Use constant factor? Where is constant factor?

- [ ] How to use sim3 in mapping
- [ ] In localization, does the descriptor idea works?
- [ ] CUrrent with the code we can only generate 3 states. How to generate 12 states? 

- [ ] What kind of descriptor should be stored in the map?

### Progress
- [x] Refactor 
	- [x] trajectory_estimator 
- [x] Create new readme for `atrium_control`, `sfm`, `test`, and `ESP8266`
- [x] Trajectory estimator use trajectory input instead of pose
- [x] Add trajectory assert equal, for landmark projection and data_association

- [x] PR

- [ ] add sim3 generator with points 
- [x] sim3 s Improvement
- [ ] sim3 unittest
- [x] sim2 Unittest 
	- [x] test 1 correct
	- [x] test 2 wrong p2 should be (10,20) not (20,20)

- [ ] Presentation
- [ ] Collect New Data