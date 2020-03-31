"""This is the Atrium Controller Generater class."""
import numpy as np
import cv2

def trajectory_estimator(image, old_estimate_trajectory, atrium_map):
    """Trajectory Estimator is a function based on iSAM to generate the estimate trajectory and the estimate state

    Args:
        image -- (H,W) numpy array of intensities, H is image height and W is image width
        old_estimate_trajectory -- (N,6) numpy array of gtsam.pose3 values
        atrium_map -- (N,3) numpy array of gtsam.point3 values

    Returns:
        new_estimate_trajectory -- (N,6) numpy array of gtsam.pose3 values
        estimate_state -- (1,12) numpy array of [x,y,z,row,pitch,yaw,dx,dy,dz,d(row),d(pitch),d(yaw)]
    """
    new_estimate_trajectory = 0
    estimate_state = 0
    new_atrium_map = 0
    return new_estimate_trajectory, estimate_state


def trajectory_generator(new_estimate_trajectory, cost_function_parameter):
    """Trajectory Generator

    Args:
        new_estimate_trajectory -- (N,6) numpy array of gtsam.pose3 values
        cost_function_parameter:

    Returns:
        desired_state:
    """

    desired_state = 1
    return desired_state


def command_generator(estimate_state, desired_state):
    """Command Generator

    Args:
        estimate_state:
        desired_state:
    
    Returns:
        actual_thrust_controller:
        actual_altitude_controller:
    """
    actual_thrust_controller = 0
    actual_altitude_controller = 0
    return actual_thrust_controller, actual_altitude_controller


class AtriumController(object):
    """Summary of the class.

    Attributes:

    """
    def __init__(self):
        self.cost_function_parameter_1 = 1
        self.cost_function_parameter_2 = 2
        self.cost_function_parameter_3 = 3

    def atrium_controller(self, image_1, image_2, image_3, old_estimate_trajectory_1, old_estimate_trajectory_2, old_estimate_trajectory_3, atrium_map):
        """Atrium Controller
        
        Args:
            image_1, image_2, image_3: (H,W) numpy array of intensities, H is image height and W is image width
            old_estimate_trajectory_1, old_estimate_trajectory_2, old_estimate_trajectory_2: (N,6) numpy array of gtsam.pose3 values
            atrium_map: (N,3) numpy array of gtsam.point3 values

        Returns:
            actual_command_1, actual_command_2, actual_command_3: (1,2) numpy array of [actual_thrust_controller, actual_altitude_controller]
                                                                  Both values are float type, they are the voltage values to control the thrust
                                                                  controller and the altitude controller motors  
        """
        # Run Trajectory Estimator Function for each blimp.
        new_estimate_trajectory_1, estimate_state1 = trajectory_estimator(
            image_1, old_estimate_trajectory_1, atrium_map)
        new_estimate_trajectory_2, estimate_state2 = trajectory_estimator(
            image_2, old_estimate_trajectory_2, atrium_map)
        new_estimate_trajectory_3, estimate_state3 = trajectory_estimator(
            image_3, old_estimate_trajectory_3, atrium_map)

        # Run Trajectory Generator Function for each blimp.
        desired_state_1 = trajectory_generator(
            new_estimate_trajectory_1, self.cost_function_parameter_1)
        desired_state_2 = trajectory_generator(
            new_estimate_trajectory_2, self.cost_function_parameter_2)
        desired_state_3 = trajectory_generator(
            new_estimate_trajectory_3, self.cost_function_parameter_3)

        # Run Command Generator Function for each blimp.
        actual_thrust_controller_1, actual_altitude_controller_1 = command_generator(
            estimate_state1, desired_state_1)
        actual_thrust_controller_1, actual_altitude_controller_2 = command_generator(
            estimate_state2, desired_state_2)
        actual_thrust_controller_1, actual_altitude_controller_3 = command_generator(
            estimate_state3, desired_state_3)

        actual_thrust_controller_1 = 1
        actual_altitude_controller_1 = 0
        actual_thrust_controller_2 = 2
        actual_altitude_controller_2 = 1
        actual_thrust_controller_3 = 3
        actual_altitude_controller_3 = 2
        actual_command_1 = np.array(
            [actual_thrust_controller_1, actual_altitude_controller_1])
        actual_command_2 = np.array(
            [actual_thrust_controller_2, actual_altitude_controller_2])
        actual_command_3 = np.array(
            [actual_thrust_controller_3, actual_altitude_controller_3])
        return actual_command_1, actual_command_2, actual_command_3

    # This is a basic function to test the script running
    def simple_atrium_controller(self):
        self.actual_thrust_controller = 1
        self.actual_altitude_controller = 0
        return self.actual_thrust_controller, self.actual_altitude_controller
