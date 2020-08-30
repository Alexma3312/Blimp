import csv
from gtsam import Point3, Rot3, Pose3
from utilities.plotting import plot_trajectory

def read_csv(file_path):
    with open(file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        poses = []
        for i,row in enumerate(readCSV):
            if i == 0:
                continue
            x = float(row[1])
            y = float(row[2])
            z = float(row[3])
            rw = float(row[4])
            rx = float(row[5])
            ry = float(row[6])
            rz = float(row[7])
            t = Point3(x,y,z)
            R = Rot3.Quaternion(rw,rx,ry,rz)
            pose = Pose3(R,t)
            poses.append(pose)

    return poses



file_path = "/home/sma96/datasets/EuRoC/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv" 
poses = read_csv(file_path)
plot_trajectory(poses, hold=True)