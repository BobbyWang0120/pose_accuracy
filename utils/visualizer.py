import cv2
import mediapipe as mp


class PoseVisualizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def draw_landmarks(self, image, results):
        """绘制骨骼点和连接线"""
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

    def draw_angles(self, image, angles):
        """在图像上显示角度信息"""
        y_pos = 30
        for joint, angle in angles.items():
            if angle is None:
                continue

            if joint == "Face Direction":
                direction = "Left" if angle > 5 else "Right" if angle < -5 else "Center"
                text = f"Face Direction: {direction} ({angle:.1f} deg)"
                color = (0, 255, 0) if abs(angle) < 5 else (0, 165, 255)
            elif "Raw" in joint:  # 原始角度显示为白色
                text = f"{joint}: {angle:.1f} deg"
                color = (255, 255, 255)
            elif "Compensated" in joint:  # 补偿后的角度
                face_angle = angles.get("Face Direction", 0)
                is_right_arm = "Right" in joint
                reliability = self.get_angle_reliability(face_angle, is_right_arm)

                color = self.get_reliability_color(reliability)
                text = f"{joint}: {angle:.1f} deg ({reliability:.0%} confidence)"
            else:
                text = f"{joint}: {angle:.1f} deg"
                color = (255, 255, 255)

            cv2.putText(image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            y_pos += 30

    def get_angle_reliability(self, face_angle, is_right_arm):
        """计算角度可信度"""
        if face_angle is None:
            return 0.5

        # 将面部角度转换为0-1之间的可信度
        if is_right_arm:
            # 右臂在面部向左时更准确
            reliability = (face_angle + 90) / 180
        else:
            # 左臂在面部向右时更准确
            reliability = (-face_angle + 90) / 180

        return min(max(reliability, 0.3), 1.0)  # 限制在30%-100%之间

    def get_reliability_color(self, reliability):
        """根据可信度返回颜色"""
        if reliability >= 0.8:
            return (0, 255, 0)  # 绿色
        elif reliability >= 0.5:
            return (0, 165, 255)  # 橙色
        else:
            return (0, 0, 255)  # 红色
