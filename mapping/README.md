# Mapping

## Modules
1. Feature Matcher
    - Image Undistortion
    - Feature Extraction
    - Feature Matching
2. Bundle Adjustment
    - Matching Tree - DSF (disjoint set forests)
    - initial estimation
    - bundle adjustment

## Module Output Diagram
    - Image Undistortion - Output Undistorted Image
    - Feature Extraction - (Superpoint, Orb)- Image with Extracted Features, feature data(.key) 
    - Feature Matching - (FLANN+RANSAC, Two Way NN) - Image with Matched Features, match data(.dat)
    - Matching Tree - 
    - initial estimation - you can change the setting in pose estimation generator to view your initial                                 estimation
    - Bundle adjustment - display Result, create result data

## Folder Organization