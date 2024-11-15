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

    def calculate_elbow_angle_with_compensation(self, landmarks, is_right_arm, face_angle):
        """计算手肘角度并根据面部朝向进行补偿"""
        if is_right_arm:
            shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        else:
            shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]

        # 基础角度计算
        base_angle = self.calculate_angle(shoulder, elbow, wrist)

        # 面部朝向补偿
        face_factor = abs(face_angle) / 90.0  # 归一化面部角度

        # 计算Z轴补偿
        # 使用肩部和手肘的相对位置来估计深度
        shoulder_width = abs(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
            - landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
        )

        # 计算投影比例
        projection_ratio = self.calculate_projection_ratio(shoulder, elbow, wrist, shoulder_width)

        # 在正面视角时进行补偿
        if abs(face_angle) < 45:  # 正面视角（±45度）
            # 使用余弦定理进行3D角度补偿
            compensation_factor = 1.0 / max(abs(np.cos(np.radians(face_angle))), 0.5)
            depth_factor = 1.0 + (projection_ratio - 1.0) * (1.0 - face_factor)

            # 根据解剖学约束调整补偿
            if base_angle < 90:
                compensated_angle = base_angle * compensation_factor * depth_factor
            else:
                compensated_angle = base_angle + (180 - base_angle) * (compensation_factor - 1)

            # 限制补偿范围，防止过度补偿
            max_compensation = 45  # 最大补偿角度
            compensation_amount = abs(compensated_angle - base_angle)
            if compensation_amount > max_compensation:
                if compensated_angle > base_angle:
                    compensated_angle = base_angle + max_compensation
                else:
                    compensated_angle = base_angle - max_compensation
        else:
            # 侧面视角时保持原始角度
            compensated_angle = base_angle

        return base_angle, compensated_angle

    def calculate_projection_ratio(self, shoulder, elbow, wrist, shoulder_width):
        """计算投影比例以估计深度影响"""
        # 计算上臂和前臂的投影长度
        upper_arm_length = np.sqrt((elbow.x - shoulder.x) ** 2 + (elbow.y - shoulder.y) ** 2)
        forearm_length = np.sqrt((wrist.x - elbow.x) ** 2 + (wrist.y - elbow.y) ** 2)

        # 使用肩宽作为参考比例
        standard_ratio = 0.7  # 标准上臂长度与肩宽的比例
        current_ratio = upper_arm_length / shoulder_width

        return current_ratio / standard_ratio

    def calculate_depth_compensation(self, shoulder, elbow, wrist, face_angle):
        """计算深度补偿因子"""
        # 计算肢体部分在图像平面上的投影长度
        upper_arm_length = np.sqrt((elbow.x - shoulder.x) ** 2 + (elbow.y - shoulder.y) ** 2)
        forearm_length = np.sqrt((wrist.x - elbow.x) ** 2 + (wrist.y - elbow.y) ** 2)

        # 根据人体解剖学标准比例进行补偿
        # 标准上臂和前臂长度比例约为1.2:1
        expected_ratio = 1.2
        actual_ratio = upper_arm_length / forearm_length if forearm_length > 0 else 1.0

        # 计算补偿系数
        ratio_diff = abs(expected_ratio - actual_ratio)
        compensation = min(ratio_diff * 0.2, 0.3)  # 限制最大补偿范围

        return compensation

    def get_fitness_angles(self, landmarks):
        """获取健身相关的关键角度"""
        if not landmarks:
            return {}

        face_angle = self.calculate_face_direction(landmarks)

        # 获取原始角度和补偿后的角度
        right_base, right_comp = self.calculate_elbow_angle_with_compensation(landmarks, True, face_angle)
        left_base, left_comp = self.calculate_elbow_angle_with_compensation(landmarks, False, face_angle)

        angles = {
            "Right Elbow Raw": right_base,
            "Right Elbow Compensated": right_comp,
            "Left Elbow Raw": left_base,
            "Left Elbow Compensated": left_comp,
            "Right Knee": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value],
            ),
            "Left Knee": self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
            ),
            "Face Direction": face_angle,
        }

        return angles
