"""
 * @file   shonan_wrapper.py
 * @date   September 2019
 * @author Shicong Ma
 * @brief  Utilities
"""
import gtsam
from gtsam import Pose3, Rot3, Point3
import os
import numpy as np


def get_g2o(filename):

    pass


def plot_g2o():
    pass


def save_g2o():
    pass


rot_0 =  Rot3.Quaternion(1, 0, 0, 0)
rot_1 = Rot3.Quaternion(0.9071908, 0.3171845, -0.2366641, 0.1427899)
rot_2 = Rot3.Quaternion(0.0433426, 0.3990360, -0.1862907, -0.8967650)
rot_3 = Rot3.Quaternion(0.1078076, -0.0946935, 0.8516455, -0.5040938)
rot_4 = Rot3.Quaternion(0.8184104, -0.2025126, 0.0306155, -0.5368945)
rot_5 = Rot3.Quaternion(0.0051805, 0.2648076, 0.3972635, 0.8786534)
rot_6 = Rot3.Quaternion(0.1572895, 0.6065701, -0.6659431, 0.4047869)
rot_7 = Rot3.Quaternion(0.3313529, 0.5611840, -0.2302828, 0.7226670)
rot_8 = Rot3.Quaternion(0.4754444, 0.7067708, -0.4274800, 0.3028011)

rot_0_1 = Rot3.Quaternion(0.9071908, 0.3171845, -0.2366641, 0.1427899)
rot_1_2 = Rot3.Quaternion(0.0819273, 0.1094217, -0.5001618, -0.8550748)
rot_2_3 = Rot3.Quaternion(0.2602866, -0.9047572, -0.2290733, -0.2473674)
rot_3_4 = Rot3.Quaternion(0.4041262, 0.4974765, -0.7449399, 0.1851044)
rot_4_5 = Rot3.Quaternion(0.5089689, 0.0224185, -0.2892013, -0.8104386)
rot_5_6 = Rot3.Quaternion(0.2525518, -0.7844494, -0.4917095, 0.2812090)
rot_6_7 = Rot3.Quaternion(0.8383972, 0.2753192, 0.3956294, -0.2544933)
rot_7_8 = Rot3.Quaternion(0.8714340, -0.2718170, -0.3729929, -0.1661162)
rot_1_8 = Rot3.Quaternion(0.6226836, 0.4615956, 0.1481179, 0.6142114)
rot_3_6 = Rot3.Quaternion(0.6766678, -0.0650692, 0.0574151, -0.7311567)
rot_7_2 = Rot3.Quaternion(0.5962602, -0.0751329, 0.7634717, 0.2365160)


# print("0",rot_0.quaternion())

# print("1",rot_1.quaternion())
# print(rot_0.compose(rot_0_1).quaternion())

print("2",rot_2.matrix())
print(rot_1.compose(rot_1_2).matrix())
print(rot_7.compose(rot_7_2).matrix())
print(rot_7.between(rot_2).quaternion())

# print("3",rot_3.quaternion())
# print(rot_2.compose(rot_2_3).quaternion())

# print("4",rot_4.quaternion())
# print(rot_3.compose(rot_3_4).quaternion())

# print("5",rot_5.quaternion())
# print(rot_4.compose(rot_4_5).quaternion())

print("6",rot_6.matrix())
print(rot_5.compose(rot_5_6).matrix())
print(rot_3.compose(rot_3_6).matrix())
print(rot_3.between(rot_6).quaternion())


# print("7",rot_7.quaternion())
# print(rot_6.compose(rot_6_7).quaternion())

print("8",rot_8.matrix())
print(rot_7.compose(rot_7_8).matrix())
print(rot_1.compose(rot_1_8).matrix())
print(type(rot_1.between(rot_8).quaternion()))
