# Data

## 1. Collect Data

The data have 11 poses.

Frame 1,2,3,4,5,6 are for mapping.
While Frame 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6 are for tracking.

## 2. Manually Calculate Ground Truth

### 2.1 First pose at the World Origin 

The poses ground truth value if the first pose is the origin of the world.


### 2.2 First pose is not at the World Origin 

The poses ground truth value if the ground under the first pose is the origin of the world. (East - x, N - y, Up - z)

- How to calculate the current coordinate system rotation matrix value to the world coordinate system(wWc)?

~~~
[[Xw],[Yw],[Zw]] = [e1_the_x_axis_vector, e2_the_y_axis_vector, e3_the_z_axis_vector][[Xc],[Yc],[Zc]]
~~~

Xw and Zc is at the same direction, Xw is on the left hand side of Zc of 30 degree.
Yw and Xc is at the same direction, Yw is on the left hand side of -Xc of 30 degree.
Yc = -Zw

~~~
theta = np.radians(30)
wRc = gtsam.Rot3(np.array([
                            [-math.sin(theta), 0 , math.cos(theta)],
                            [-math.cos(theta), 0 , -math.sin(theta)],
                            [0, 1 , 0]
                        ])
z = h = 1.2
tc = 1.58
translation [x,y] = i*[-math.sin(theta)*1.58,-math.cos(theta)*1.58]
~~~

## 3. Run Superpoint and create feature data files

- [x] Add front end functions to save matched information between every two frames.

## 4. Manually create matched feature points for mapping

- The track_test.cvs in Blimps/SuperPointPretrainedNetwork/dataset/key_points
saved the feature tracking information
    - this .cvs file is created in demo_superpoint draw_tracks()

~~~
self.Z = [
                [Point2(548,248),Point2(228,252),Point2(184,264),Point2(368,440),Point2(548,288),Point2(328,400),Point2(144,244),Point2(216,384),Point2(212,324),Point2(168,368)],
                [Point2(536,228),Point2(216,236),Point2(180,244),Point2(360,424),Point2(536,268),Point2(316,384),Point2(136,228),Point2(204,368),Point2(200,308),Point2(156,352)],
                [Point2(492,236),Point2(176,244),Point2(140,248),Point2(312,432),Point2(492,276),Point2(268,392),Point2(96,236),Point2(160,368),Point2(156,320),Point2(112,352)],
                [Point2(492,228),Point2(180,244),Point2(148,244),Point2(316,428),Point2(492,272),Point2(276,388),Point2(104,236),Point2(164,360),Point2(164,312),Point2(116,348)],
                [Point2(448,236),Point2(152,244),Point2(124,252),Point2(280,432),Point2(444,280),Point2(240,388),Point2(80,240),Point2(132,364),Point2(128,312),Point2(80,344)]
            ]

self.Z = [
                [Point2(548/4,248/4),Point2(228/4,252/4),Point2(184/4,264/4),Point2(368/4,440/4),Point2(548/4,288/4),Point2(328/4,400/4),Point2(144/4,244/4),Point2(216/4,384/4),Point2(212/4,324/4),Point2(168/4,368/4)],
                [Point2(536/4,228/4),Point2(216/4,236/4),Point2(180/4,244/4),Point2(360/4,424/4),Point2(536/4,268/4),Point2(316/4,384/4),Point2(136/4,228/4),Point2(204/4,368/4),Point2(200/4,308/4),Point2(156/4,352/4)],
                [Point2(492/4,236/4),Point2(176/4,244/4),Point2(140/4,248/4),Point2(312/4,432/4),Point2(492/4,276/4),Point2(268/4,392/4),Point2(96/4,236/4),Point2(160/4,368/4),Point2(156/4,320/4),Point2(112/4,352/4)],
                [Point2(492/4,228/4),Point2(180/4,244/4),Point2(148/4,244/4),Point2(316/4,428/4),Point2(492/4,272/4),Point2(276/4,388/4),Point2(104/4,236/4),Point2(164/4,360/4),Point2(164/4,312/4),Point2(116/4,348/4)],
                [Point2(448/4,236/4),Point2(152/4,244/4),Point2(124/4,252/4),Point2(280/4,432/4),Point2(444/4,280/4),Point2(240/4,388/4),Point2(80/4,240/4),Point2(132/4,364/4),Point2(128/4,312/4),Point2(80/4,344/4)]
            ]

~~~

## 5. Manually create landmark descriptors for pose estimation

Landmark 1:

- All descriptors:

- Average:

- Normalize:
# Compute the norm, then divide by norm to normalize.
