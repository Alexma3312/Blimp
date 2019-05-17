# DataSet

## Naming Convention

- Input image frames and director

## 4.3_Atrium_dataset

This folder will contain both mapping dataset and tracking data set.

### Mapping Dataset

This contains 
    - frames 
        - **Important** `Naming conventions` for the frames
    - frames with extracted features
    - a document of the group truth poses of the frames
    - a document of matched feature information of every two frames (i_j)
    - a document of manually matched features

### Tracking Dataset

This contains
    - frames (very close to each other, obtain certain features of the map)
    - frames with extracted features
    - a document of the group truth poses of the frames
    - a document of manually matched features
    - a document of matched feature information of every adjacent three frames (i_i+1_i+2)
