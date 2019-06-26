# Mapping Process Tutorial

Follow the steps below to perform SfM.

## 1. Collect data

### 1.1 Charge battery

### 1.2 Test Camera and receiver with VLC(ubuntu app)

### 1.3 Run script to collect data

Go to `/home/sma96/Blimps/datasets/scripts`    
Run `python_video_test.py` to collect data
Move data to a dataset folder. Create a new folder `source_images` and place the images under `source_images` folder.

## 2. Calibration

Run `run_undistort.py` to undistort images.

- set the image directory path in the script

## 3. Superpoint feature extraction

Run `run_feature_extraction.py` to generate `.key` files.

- set the image directory path in the script

## 4. Feature Matching (data association)

1. Go to run `4d-agriculture` repository

2. Create a folder in 4d-agriculture/dataset

3. Create `input` and `output` folders in the folder

- 4d-agriculture
    - dataset
        - input
        - output

4. Copy `.key` and `frame_.jpg` files to `input` folder

5. Skip get match when there are not enough point correspondences for RANSAC

~~~
Add `break` in FeatureMatcher.cpp/void FeatureMatcher::getMatches()
~~~

6. Run `/4d-agriculture/cpp/exe/Shicong04_LibraryDataMatcher.cpp` to generate 
    - feature_.dat
    - match_.dat

## 5. landmark correspondence

1. Copy `_.key` and `match_.dat` to `Blimps` repository dataset `root` folder

2. Copy `match.jpg` to `Blimps` repository dataset `match_images` folder
