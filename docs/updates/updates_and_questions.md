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
- [ ] data association unittest

