# Structure from Motion (SfM)

## Notes:
1. Import problem
- The problem is caused by the difference of the `root paths`.
	- If code is ran in a terminal, it is relative path, the root path starts at the current path and need to add `../` to search the parent directory.      
    The line `sys.path.append('../')` should be added before import.      
    This line will change location when doing `sort import` or `format document`.
	- If code is ran in VScode, it is also relative path but the root path starts at the `Blimps` 

## Overview

SfM includes:

1. Source & Test

- `sfm/atrium_sfm.py`: SfM GTSAM solution module
- `tests/test_atrium_sfms.py`: module test of `atrium_sfm` 

2. Data

-  `sfm/sfm_data.py`: Create SfM test data

3. Similarity Transfrom

- `sfm/sim2.py`: Similarity 2 transform
- `sfm/sim3.py`: Similarity 3 transform
