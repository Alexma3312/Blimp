"""Back Projection"""
# pylint: disable=no-name-in-module,no-member, invalid-name
from gtsam import Point3, Point2, Pose3


def back_projection(calibration, key_point=Point2(), pose=Pose3(), depth=20):
    """
    Back Projection Function.
    Input:
        key_point-gtsam.Point2, key point location within the image.
        pose-gtsam.Pose3, camera pose in world coordinate.
    Output:
        gtsam.Pose3, landmark pose in world coordinate.
    """
    # Normalize input key_point
    normalized_point = calibration.calibrate(key_point)
    # Transfer normalized key_point into homogeneous coordinate and scale with depth
    homogeneous_point = Point3(
        depth*normalized_point.x(), depth*normalized_point.y(), depth)
    # Transfer the point into the world coordinate
    return pose.transform_from(homogeneous_point)
