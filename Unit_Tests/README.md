# Atrium Controller unit test
Inputs: 
- 3 images
- 3 estimate trajectory till state t-1
- The MAP

Outputs:
- 3 Commands

# Commmand Generator unit test
Inputs:
- 1 estimate pose/vel at state t
- 1 desired pose/vel at state t+1

Outputs:
- 1 command

# Pose Generator unit test
Inputs:
- 3 estimate trajectory up to the state t
- [cost function pararmeter]

Output:
- 3 desired pose/vel at state t+1

# Trajectory Generator unit test
Inputs:
- estimate traj t-1
- input image
- The MAP

Outputs:
- estimate trajectory till state t
- estimate pose/vel at state t





