# Calibration

## Matlab Calibration

### R2018a not recommend
- http://www.vision.caltech.edu/bouguetj/calib_doc/
- This calibration method is not recommended because the process is complicate and cumbersome, which you have to take 20 images and manually Mark out the chessboard.

- Little thing about running **matlab with license** on ubuntu
    - Check hostid, login name through 
        1. /usr/local/MATLAB/R2018a/bin$ sudo ./activate_matlab.sh 
        2. activate without internet    
        3. I do not have a license help me with the next step.
        4. fill in https://www.mathworks.com/licensecenter/licenses/621625/2075860/activations 

### R2019a highly recommended
- computer vision toolbox


## Boofcv Calibration
- https://boofcv.org/index.php?title=Tutorial_Camera_Calibration
- Based on Java.
- The application is easy to download and easy to use.
- I downloaded and run the application but the problem is that the application can not recognized my web camera, therefore I gave up using this method.

## ROS Python Calibration
- http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration
- Higher Recommended
- But required some basic ROS knowledge.

### ROS Python Process
- Follow ROS wiki to download ROS and create a workspace
- Get camera output through ROS web camera drive, `usb_cam` node: http://wiki.ros.org/usb_cam
- roscd usb_cam and modify the launch file parameters or modify the parameter through the roslaunch command (e.g. video_device:=/dev/video1, autofocus:= false, pixel_format:=mjpeg)
    - `/dev/video0` into `/dev/video1`
    - `yuyv` into `mjpeg`
    -  `autofocus` `false`

- rosrun usb_cam_node usb_cam
- The output topic should be 
    - /usb_cam/camera_info
    - /usb_cam/image_raw

- source /opt/ros/kinetic/setup.bash
- source ~/catkin_ws/devel/setup.bash
- roscore
- If you are using the chessboard from the rail lab:
    - The command is       
    rosrun camera_calibration cameracalibrator.py --size 9x7 --square 0.06985 image:=/usb_cam/image_raw camera:=/usb_cam
0.06985
0.0635
### The First Result
[image]

width
640

height
480

[narrow_stereo]

camera matrix
343.555173 0.000000 295.979699
0.000000 327.221818 261.530851
0.000000 0.000000 1.000000

distortion
-0.305247 0.064438 -0.007641 0.006581 0.000000

rectification
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000

projection
235.686951 0.000000 304.638818 0.000000
0.000000 256.651520 265.858792 0.000000
0.000000 0.000000 1.000000 0.000000

### The Second Result
[image]

width
640

height
480

[narrow_stereo]

camera matrix
333.383915 0.000000 303.574395
0.000000 314.669739 247.635818
0.000000 0.000000 1.000000

distortion
-0.282548 0.054412 -0.001882 0.004796 0.000000

rectification
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000

projection
226.994629 0.000000 311.982613 0.000000
0.000000 245.874146 250.410089 0.000000
0.000000 0.000000 1.000000 0.000000