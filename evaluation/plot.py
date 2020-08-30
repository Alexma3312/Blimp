"""A visualization script based on colmap read_model.py.
For Colmap Images.txt, Sesync poses.txt, and shonan shonan.dat results visualization.
"""
import collections
import math
import os
import struct
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Sesync = collections.namedtuple(
    "Sysync", ["id", "R", "t"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


class SEsync(Sesync):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                # if t is the translation in camera coordinate
                R = qvec2rotmat(qvec)
                tvec = - R.T.dot(tvec)
                qvec = rotmat2qvec(R.T)
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_shonan(path):
    """
    read dat file
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                id = int(elems[0])
                R = np.array(elems[1:]).astype(float).reshape(3, 3)
                assert(np.allclose(np.zeros((3, 3)), abs(
                    R.dot(R.T) - np.eye(3)), atol=1e-4))
                images[id] = R
    return images


def read_sesync(path):
    """
    read sesync generated pose.txt file
    """
    images = {}
    with open(path, "r") as fid:
        result = []
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                result.append(tuple(map(float, elems)))
        result = np.array(result)
        num_poses = int(result.shape[1]/4)
        ts = result[:, :num_poses]
        Rs = result[:, num_poses:]
        for id in range(num_poses):
            t = ts[:, id]
            R = Rs[:, id*3: id*3+3]
            # t = -R.T.dot(t)
            assert(np.allclose(np.zeros((3, 3)), abs(
                R.dot(R.T) - np.eye(3)), atol=1e-4))
            images[id] = SEsync(id=id, R=R, t=t)
    return images


def read_model(path, ext):
    """Read poses information from files."""
    if ext == "colmap":
        images = read_images_text(os.path.join(path, "images.txt"))
    elif ext == "sesync":
        images = read_sesync(os.path.join(path, "poses.txt"))
    elif ext == "shonan":
        # index = path.find("visualize_shonan") + 17
        # name = path[index:]
        # file = "shonan_result_of_" + name + ".dat"
        images = read_shonan(os.path.join(
            path, "shonan_result_of_shicong.dat"))
    else:
        images = read_images_binary(os.path.join(path, "images" + ext))
    return images


def qvec2rotmat(qvec):
    """Quaternion vector to rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    """Rotation matrix to quaternion vector."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def plot_axes(t, R, axes, r, g, b, axis_length=0.15):
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


def plot_pose(ts, Rs):
    """
        @params ts: translation of camera poses, N list
        @params Rs: rotation of camera poses, N list 
    """
    fig = plt.figure()
    axes = fig.gca(projection='3d')
    axis_length = 0.15
    for i in range(len(ts)):
        t = ts[i] - ts[0]
        R = Rs[i]
        assert(math.isclose(np.linalg.norm(R[:, 1]) + np.linalg.norm(R[:, 1]) + np.linalg.norm(R[:, 1]),
                            3, rel_tol=1e-2))
        # draw the camera axes
        plot_axes(t, R, axes, 'r-', 'g-', 'b-', axis_length=axis_length)


def plot_origin(axes, axis_length=0.15):
    """Plot the coordinate system origin."""
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t = np.array([0, 0, 0])
    plot_axes(t, R, axes, 'r-', 'g-', 'b-', axis_length=axis_length)


def plot_normalized_poses(rtc, rRc, wtc=[], wRc=[]):
    """
        @params rtc: camera to result coordinate translation of camera poses, N list
        @params rRc: camera to result coordinate rotation of camera poses, N list 
    """
    fig = plt.figure()
    axes = fig.gca(projection='3d')
    axis_length = 0.15

    # w stands for world and r stands for result
    wR0 = wRc[0]
    rR0 = rRc[0]
    wRr = wR0.dot(rR0.T)

    wt0 = wtc[0]
    rt0 = rtc[0]
    wtr = wt0-rt0

    for i in range(len(rtc)):
        rti = rtc[i]
        t = wtr + rti
        rRi = rRc[i]
        R = wRr.dot(rRi)
        assert(math.isclose(np.linalg.norm(R[:, 1]) + np.linalg.norm(R[:, 1]) + np.linalg.norm(R[:, 1]),
                            3, rel_tol=1e-2))
        # draw the camera axes
        plot_axes(t, R, axes, 'r-', 'g-', 'b-', axis_length=axis_length)

    plot_origin(axes, axis_length=0.15)


def plot_poses_comparision(rtc, rRc, wtc, wRc):
    """
        @params rtc: camera to result coordinate translation of camera poses, N list
        @params rRc: camera to result coordinate rotation of camera poses, N list 
        @params wtc: camera to world coordinate translation of camera poses, N list
        @params wRc: camera to world coordinate rotation of camera poses, N list 
    """
    fig = plt.figure('Results', constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axis_length = 0.15
    # w stands for world frame and r stands for result frame
    wR0 = wRc[0]
    rR0 = rRc[0]
    wRr = wR0.dot(rR0.T)

    wt0 = wtc[0]
    rt0 = rtc[0]
    wtr = wt0-rt0

    # Plot colmap result
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    ax.set_title('Colmap')
    for i in range(len(wtc)):
        t = wtc[i]
        R = wRc[i]
        assert(math.isclose(np.linalg.norm(R[:, 1]) + np.linalg.norm(R[:, 1]) + np.linalg.norm(R[:, 1]),
                            3, rel_tol=1e-2))
        # draw the camera axes
        plot_axes(t, R, ax, 'r-', 'g-', 'b-', axis_length=axis_length)
    plot_origin(ax, axis_length=0.15)

    # Plot shonan result
    ax = fig.add_subplot(gs[0, 1], projection='3d')
    ax.set_title('Shonan')

    for i in range(len(rtc)):
        rti = rtc[i]
        t = wtr + rti
        rRi = rRc[i]
        R = wRr.dot(rRi)
        assert(math.isclose(np.linalg.norm(R[:, 1]) + np.linalg.norm(R[:, 1]) + np.linalg.norm(R[:, 1]),
                            3, rel_tol=1e-2))
        # draw the camera axes
        plot_axes(t, R, ax, 'r-', 'g-', 'b-', axis_length=axis_length)
    plot_origin(ax, axis_length=0.15)

    # Plot alignment result
    ax = fig.add_subplot(gs[1, :], projection='3d')
    ax.set_title('Alignment')
    for i in range(len(rtc)):
        t = wtc[i]
        R = wRc[i]
        assert(math.isclose(np.linalg.norm(R[:, 1]) + np.linalg.norm(R[:, 1]) + np.linalg.norm(R[:, 1]),
                            3, rel_tol=1e-2))
        # draw the camera axes
        plot_axes(t, R, ax, 'g-', 'g-', 'g-', axis_length=axis_length)

        rti = rtc[i]
        t = wtr + rti
        rRi = rRc[i]
        R = wRr.dot(rRi)
        assert(math.isclose(np.linalg.norm(R[:, 1]) + np.linalg.norm(R[:, 1]) + np.linalg.norm(R[:, 1]),
                            3, rel_tol=1e-2))
        # draw the camera axes
        plot_axes(t, R, ax, 'r--', 'r--', 'r--', axis_length=axis_length)
    plot_origin(ax, axis_length=0.15)


def main():
    """Execution."""
    if len(sys.argv) != 5:
        print("Usage: python visualize_camera_pose.py path/to/model/folder colmap sesync shonan")
        return

    ts = []
    if (sys.argv[2] == "colmap"):
        images = read_model(path=sys.argv[1], ext=sys.argv[2])
        Rs = []
        for _, k in enumerate(images):
            ts.append(images[k].tvec)
            Rs.append(qvec2rotmat(images[k].qvec))
        # print("Showing Colmap Result")
        # plot_pose(ts, Rs)
    sesync_ts = []
    if (sys.argv[3] == "sesync"):
        sesync_result = read_model(path=sys.argv[1], ext=sys.argv[3])
        sesync_Rs = []
        for _, k in sesync_result.items():
            sesync_ts.append(k.t)
            sesync_Rs.append(k.R)
        print("Showing Sesync Result")
        plot_normalized_poses(sesync_ts, sesync_Rs, ts, Rs)
    if (sys.argv[4] == "shonan"):
        shonan_result = read_model(path=sys.argv[1], ext=sys.argv[4])
        shonan_Rs = list(shonan_result.values())
        print("Showing Shonan and Colmap Results")
        shonan_ts = ts
        plot_poses_comparision(shonan_ts, shonan_Rs, ts, Rs)
    plt.show()

    # print("num_images:", len(images))


if __name__ == "__main__":
    main()