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

    def calculate_face_direction(self, landmarks):
        """Calculate face direction angle relative to camera"""
        if not landmarks:
            return None

        # 获取面部关键点
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE.value]

        # 计算眼睛之间的距离（在2D图像上的投影）
        eye_distance = np.sqrt((right_eye.x - left_eye.x) ** 2 + (right_eye.y - left_eye.y) ** 2)

        # 计算鼻子相对于眼睛中点的位置
        eye_center_x = (left_eye.x + right_eye.x) / 2
        eye_center_y = (left_eye.y + right_eye.y) / 2

        # 计算朝向角度
        # 使用眼睛距离作为参考来估计深度
        face_angle = np.arctan2(nose.x - eye_center_x, eye_distance) * 180 / np.pi

        return face_angle

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
            # Add face direction
            "Face Direction": self.calculate_face_direction(landmarks),
        }

        return angles
