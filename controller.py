# controller.py 

import numpy as np
import cv2
import math

class FlightController:

    # STATES
    SEARCH = 0
    TRACK = 1
    DESCEND = 2
    BLIND_DESCEND = 3
    LANDED = 4

    def __init__(self):
        # Initialize your PID constants, filters, and state variables here

        # Physical Parameters
        self.mass = 1.2
        self.gravity = 9.81

        # Each motor needs approximately this much to hover
        self.hover_force = self.mass * self.gravity / 4.0

        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_6X6_50
            ),
            cv2.aruco.DetectorParameters()
        )

        self.prev_roll = 0.0
        self.prev_pitch = 0.0

    @staticmethod
    def quat_to_euler(q):
        x, y, z, w = q

        pitch = math.atan2(
            2*(w*x + y*z),
            1 - 2*(x*x + y*y)
        )

        roll = math.asin(
            max(-1, min(1,
                2*(w*y - z*x)
            ))
        )

        yaw = math.atan2(
            2*(w*z + x*y),
            1 - 2*(y*y + z*z)
        )

        return roll, pitch, yaw
        
    def calculate_forces(self, camera_frame, drone_pos, drone_ori, drone_vel):
        """
        This function is called by the arena 240 times per second.
        You must return the 4 motor forces in Newtons.
        """
        
        # [A] VISION PROCESSING (cv2.aruco to find the rover in camera_frame)

        corners, ids, rejected = self.detector.detectMarkers(camera_frame)
        print("IDs:", ids)
        print("Corners:", corners)
        if len(rejected) > 0:
            print("Rejected:", len(rejected))

        # Fallback if aruco marker not found in frame
        if ids is None:
            hover_force = self.mass * self.gravity / 4
            return hover_force, hover_force, hover_force, hover_force

        # [B] STATE ESTIMATION (Calculate error between drone_pos and rover)
        
        h, w = camera_frame.shape[:2]
        image_cx = w / 2
        image_cy = h / 2
        pts = corners[0][0]
        marker_cx = np.mean(pts[:,0])
        marker_cy = np.mean(pts[:,1])
        error_x = marker_cx - image_cx
        error_y = marker_cy - image_cy
        print(error_x, error_y)
        height = drone_pos[2]

        # [C] CONTROL THEORY (PID loop to determine thrust)
        
        roll, pitch, yaw = self.quat_to_euler(drone_ori)

        print("Roll, Pitch: ", np.degrees(roll), np.degrees(pitch))

        hover_thrust = self.mass * self.gravity
        T = hover_thrust

        kx = 1
        ky = 1
        kp_att = 10
        kd_att = 1

        normalized_x = error_x / (w / 2)
        normalized_y = error_y / (h / 2)

        cam_offset_x = height * math.tan(roll)
        cam_offset_y = height * math.tan(pitch)

        target_x = normalized_x - cam_offset_x
        target_y = normalized_y - cam_offset_y

        max_angle = np.radians(25)

        desired_roll = np.clip(3 * normalized_x, -max_angle, max_angle)
        desired_pitch = np.clip(3 * normalized_y, -max_angle, max_angle)

        print("Desired:", np.degrees(desired_roll), np.degrees(desired_pitch))
        
        roll_error  = desired_roll - roll
        pitch_error = desired_pitch - pitch

        # Finding angular velocities
        dt = 1.0 / 240.0
        roll_rate  = (roll  - self.prev_roll)  / dt
        pitch_rate = (pitch - self.prev_pitch) / dt
        self.prev_roll  = roll
        self.prev_pitch = pitch


        u_roll = - kp_att * roll_error + kd_att * roll_rate
        u_pitch = kp_att * pitch_error - kd_att * pitch_rate

        # u_roll  = kx * normalized_x
        # u_pitch = ky * normalized_y

        f1 = T/4 + u_roll + u_pitch
        f2 = T/4 - u_roll + u_pitch
        f3 = T/4 + u_roll - u_pitch
        f4 = T/4 - u_roll - u_pitch

        f1 = np.clip(f1, 0.0, 20.0)
        f2 = np.clip(f2, 0.0, 20.0)
        f3 = np.clip(f3, 0.0, 20.0)
        f4 = np.clip(f4, 0.0, 20.0)

        print("Drone Pos: ", drone_pos)
        print("Drone Vel: ", drone_vel)

        return f1, f2, f3, f4

