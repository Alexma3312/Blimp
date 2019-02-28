# Records

## Date

### Questions

### Progress

- [ ] SFM feature point association
- [ ] trajectory estimator velocity and rotation accelerations

## 2.27

### Questions

1. Will FOV change when downsampling images?
2. How should I calculate the area that landmarks can be captured
3. Because the output of Superpoint descriptors are arrays should I also use array to store feature points (2D list or 2D np.array)
4. When doing functional programming and I want to create a trajectory that also preserve the previous poses, in this case should I create a function that use old trajectory and a new pose as input, and the ouput should be a new trajectory?
~~~
def trajectory_generator():
	input:
		- past_trajectory
		- new_pose

	output:
		- new_trajectory
~~~ 

### Progresses

- [ ] Generate trajectory estimator unittest data.
- [ ] data association unittest
- [ ] trajectory_estimator unittest
- [x] study functional programming: http://www.lihaoyi.com/post/WhatsFunctionalProgrammingAllAbout.html
- [ ] use submodule to substitute superpoint


## 2.28

### Questions

### Progress