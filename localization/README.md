# Localization

## Pipeline


a. localize by matching in the map (This will not take long since most functions have been developed before)

0. Create methods to save and load the mapping result. This includes storing the normalized averaged descriptor values in the map.
1. Create class TrajectoryEstimator based on unittest
2. Create class VideoStreamer to handle both image and video input based on unittest
3. Create script to execute.
Pipeline:
~~~
    0. Create instances for TrajectoryEstimator and VideoStreamer(handle bad input)
    1. update_trajectory()
    {
        trajectory = [initial pose]
        2. next_frame()
        {
            if bad frame:
                continue
            if good frame:
                3. get_pose_from_trajectory()
                4. undistort image
                5. superpoint_extraction()
                6. landmark_projection()
                7. landmark_association()
                8. pose_estimate()
                9. plot()
                10. trajectory.append(new pose)
        }
    }
~~~
b. localize by tracking
Use matched features to estimate the camera motion(row,pitch,yaw,x,y,z,scale). 