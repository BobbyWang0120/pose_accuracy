import numpy as np
import mediapipe as mp


class AngleCalculator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def calculate_angle(self, a, b, c):
        """计算三个点形成的角度"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def get_fitness_angles(self, landmarks):
        """Get key angles for fitness analysis"""
        if not landmarks:
            return {}

        angles = {
            # Right arm angle
            "Right Elbow": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value],
            ),
            # Left arm angle
            "Left Elbow": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value],
            ),
            # Right knee angle
            "Right Knee": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value],
            ),
            # Left knee angle
            "Left Knee": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
            ),
        }

        return angles
