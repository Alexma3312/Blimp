import matplotlib.pyplot as plt
from gtsam import Point3, Pose3, Rot3
from gtsam.utils import plot
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D


from evaluation.sim3 import *

# # Create expected sim3
# expected_R = Rot3.Rz(math.radians(-90))
# expected_s = 2
# expected_t = Point3(6, 8, 10)

# # Create source points
# s_point1 = Point3(0, 0, 0)
# s_point2 = Point3(3, 0, 0)
# s_point3 = Point3(3, 0, 4)

# # Create destination points
# sim3 = Similarity3()
# sim3._R = expected_R
# sim3._t = expected_t
# sim3._s = expected_s
# d_point1 = sim3.point(s_point1)
# d_point2 = sim3.point(s_point2)
# d_point3 = sim3.point(s_point3)
# print(d_point1,d_point2,d_point3)

# fig = plt.figure(1)
# axes = fig.gca(projection='3d')
# axes.scatter(0, 0, 0, c= 'r')
# axes.scatter(3, 0, 0, c='r')
# axes.scatter(3, 0, 4, c='r')

# axes.scatter(6, 8, 10, c='g')
# axes.scatter(6, 2, 10, c='g')
# axes.scatter(6, 2, 18, c='g')
# # draw
# # x_axe = y_axe= z_axe=5
# # axes.set_xlim3d(-x_axe, x_axe)
# # axes.set_ylim3d(-y_axe, y_axe)
# # axes.set_zlim3d(-z_axe, z_axe)
# plt.pause(1)
# plt.legend()
# plt.show()


def plot_axes(t, R, axes, r, g, b, axis_length=0.5):
    """Plot axes."""
    x_axis = t + R[:, 0] * axis_length
    line = np.append(t[np.newaxis], x_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], r)

    y_axis = t + R[:, 1] * axis_length
    line = np.append(t[np.newaxis], y_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], g)

    z_axis = t + R[:, 2] * axis_length
    line = np.append(t[np.newaxis], z_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], b)


fig = plt.figure(1)
axes = fig.gca(projection='3d')

st1 = np.array([0, 0, 0])
sR1 = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
plot_axes(st1,sR1,axes,'r','r','r')

st2 = np.array([4, 0, 0])
sR2 = np.array([[-1, 0, 0],[0, 1, 0],[0, 0, 1]])
plot_axes(st2,sR2,axes,'r','r','r')

dt1 = np.array([4, 6, 10])
dR1 = np.array([[-1, 0, 0],[0, 1, 0],[0, 0, -1]])
plot_axes(dt1,dR1,axes,'g','g','g')

dt2 = np.array([-4, 6, 10])
dR2 = np.array([[1, 0, 0],[0, 1, 0],[0, 0, -1]])
plot_axes(dt2,dR2,axes,'g','g','g')




plt.pause(1)
plt.legend()
plt.show()