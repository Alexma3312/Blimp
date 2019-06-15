import unittest
import cv2
import numpy as np
import gtsam 
from gtsam import Rot3, Pose3, Point3, Point2
import math
from sfm.mapping_a_line_with_auto_corresponding import MappingFrontEnd
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt

def create_img_path():
    img_paths = []
    for i in range(25):
        for j in range(8):
            path =i+ 0.1*j
            img_paths.append(path)
    print(img_paths)
    return img_paths

def create_pose_initial_estimation(row,col,angles):
        theta = np.radians(30)
        wRc = np.array([[-math.sin(theta), 0 , math.cos(theta)],
                        [-math.cos(theta), 0 , -math.sin(theta)],
                        [0, -1 , 0]])
        delta = np.radians(45)
        R = Rot3(np.array([[math.cos(delta), 0 , math.cos(delta)],
                        [0, 1 , 0],
                        [-math.sin(delta), 0 , math.sin(delta)]]))
        delta_x = delta_y = 1.25

        row_idx = 0
        col_idx = 0
        angle_idx = 0
        angle_poses = np.array([])
        col_poses = np.array([]).reshape(0,angles)
        row_poses = np.array([]).reshape(0,col,angles)

        for i in range(row*col*angles):
            image_pose = Pose3(Rot3(np.dot(wRc,
                        # cRc'
                        np.array([[math.cos(delta*4), 0 , -math.sin(delta*4)],
                        [0, 1 , 0],
                        [math.sin(delta*4), 0 , math.cos(delta*4)]])
                        )),
                        Point3(-math.sin(theta)*delta_x*col_idx+math.cos(theta)*delta_y*row_idx,-math.cos(theta)*delta_x*col_idx-math.sin(theta)*delta_y*row_idx,1.2))
                        # Point3(-math.sin(theta)*1.58*col_idx, -math.cos(theta)*1.58*col_idx, 1.2))
            angle_poses = np.hstack((angle_poses, image_pose))            
            angle_idx += 1
            if(angle_idx == angles):
                angle_idx = 0
                col_poses = np.vstack((col_poses, np.expand_dims(angle_poses,axis=0)))
                col_idx+=1
                if(col_idx == col):
                    col_idx = 0
                    row_idx += 1
                    row_poses = np.vstack((row_poses, np.expand_dims(col_poses,axis=0)))
                    col_poses = np.array([]).reshape(0,angles)
                angle_poses = np.array([])
            

        pose_initial = row_poses

        fignum = 0

        fig = plt.figure(fignum)
        axes = fig.gca(projection='3d')
        plt.cla()

        # Plot cameras
        for i in range(row*col*angles):
            ir,i_mod = divmod(i,angles*col) 
            ic, ia = divmod(i_mod, angles)
            # print(ir,ic,ia)
            gtsam_plot.plot_pose3(fignum, pose_initial[ir][ic][0], 1)
            # gtsam_plot.plot_pose3(fignum, pose_initial[ir][ic][1], 1.4)


        # draw
        axes.set_xlim3d(-10, 10)
        axes.set_ylim3d(-10, 10)
        axes.set_zlim3d(-10, 10)
        plt.legend()
        plt.show()

class TestFrontEnd(unittest.TestCase):
    # def setUp(self):
        # self.img_paths = create_img_path()
        
             
    def test_create_pose_initial_estimation(self):
        create_pose_initial_estimation(1,5,1)

    # def assertGtsamEquals(self, actual, expected, tol=1e-2):
    #     """Helper function that prints out actual and expected if not equal."""
    #     equal = actual.equals(expected, tol)
    #     if not equal:
    #         raise self.failureException(
    #             "Not equal:\n{}!={}".format(actual, expected))
    # def test_back_projection(self):
    #     fe = MappingFrontEnd()
    #     theta = np.radians(30)
    #     wRc = Rot3(np.array([[-math.sin(theta), 0 , math.cos(theta)],
    #                     [math.cos(theta), 0 , math.sin(theta)],
    #                     [0, -1 , 0]]))

    #     expected_point1 = Point3(10*math.cos(theta),10*math.sin(theta),1.2)
    #     expected_point2 = Point3()

    #     key_point1 = Point2(320,240)
    #     pose1 = Pose3(wRc,Point3(0, 0, 1.2))

    #     key_point2 = Point2(5,5)
    #     pose2 = Pose3(wRc,Point3(-math.sin(theta)*1.58*1, -math.cos(theta)*1.58*1, 1.2))

    #     actual_point1 = fe.back_projection(key_point1,pose1)
    #     self.assertGtsamEquals(expected_point1,actual_point1)
    #     actual_point2 = fe.back_projection(key_point2,pose2)


if __name__ == "__main__":
    unittest.main()