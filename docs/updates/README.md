# Coding Dogma
1. unittest: https://docs.python.org/3/library/unittest.html
2. We will need to follow the google python style for coding: https://github.com/google/styleguide/blob/gh-pages/pyguide.md
3. Write [doc string](https://www.geeksforgeeks.org/python-docstrings) at the beginning of each script
4. Write doc string for each function including test functions.
5. Write annotation for key steps of a function.
6. Declare the function before you call it.
7. Less Abbreviations

# Import GTSAM
1. Download gtsam from:
https://bitbucket.org/gtborg/gtsam/src/08ba9d9cedeacca9cac049c98fe42dea26ad0bcf/?at=develop
2. Follow the modified the instruction from the `README.md` in the home page of `gtsam`
~~~
mkdir build
cd build
~~~
- In this step choose cython install tool box, unclick the unstable box
~~~
cmake-gui .. 
make install -j4
~~~

3. Follow the instruction in `README.md` in `cython` file
~~~
export PYTHONPATH=$PYTHONPATH:<GTSAM_CYTHON_INSTALL_PATH>
~~~

# Problems
1. import gtsam with python3
2. Call function from different class.
3. Call function from different files
4. Where to check the functions of GTSAM
5. I did not find IMU in the gtsam_visual_slam_example script

6. How to create features
7. How to calculate speed? a. derivative of pose or b. feature tracking
8. What is the field of view of the simulated camera
9. What is the frame rate
10. What is X(i), gtsam.Rot3

1. How to initialize a numpy array
2. what is the mini of len()
3. what is the output of isam, will landmark merge together


## Solved
1. the static function in class will be share by all objects
2. import will automatically create an object of the class
3. What is an up direction vector for camera
- up direction vector determine the row of the camera, pitch and yaw are determined by the target and camera eye pose

# Unit Tests Brief Description

## Arguments
- images: cv.mat
- trajectories: numpy array of Point3
- MAP: numpy array of Point3
- commands: two element array
- pose: Pose3
- vel:

## Atrium Controller unit test
Inputs: 
- 3 images
- 3 estimate trajectory till state t-1
- The MAP

Outputs:
- 3 commands

## Commmand Generator unit test
Inputs:
- **1 estimate pose/vel at state t**
- 1 desired pose/vel at state t+1

Outputs:
- 1 command

## Trajectory Generator unit test
Inputs:
- **3 estimate trajectory up to the state t**
- [cost function pararmeter]

Output:
- 3 desired pose/vel at state t+1

## Trajectory Estimator unit test
Inputs:
- estimate traj t-1
- input image
- The MAP

Outputs:
- estimate trajectory till state t
- estimate pose/vel at state t
- The new MAP

## Feature Extract
Inputs:
- input image

Outputs:
- landmarks

### More information on Trajectory Estimator
Trajectory Estimator is actually iSAM

Map is the a collection of Points (_points)
- Points is a numpy array of point3 (S03) 

Based on state, there are two types of maps:
a. State t-1 map
b. State t map

Based on generating methods, there are two types of maps
a. Map generated with single blimp data
b. Large Scale Map generated with multiple blimp data
- Large Scale Map can be generated in two ways
a. Generating map with three input images
b. Register three maps into one map

Trajectory is a collection of Poses (_poses)
- Poses is a numpy array of pose3 (SE3) 



# Questions?
1. Maybe Feature Points need to be classified?


2. Camera only provides points?

3. How is pose3 represented?
4. who provides the new pose information, iSAM or geometry by myself?
5. Where to get the vel?

6. IMU factor provides the vel?
7. Where to get the IMU information? 








