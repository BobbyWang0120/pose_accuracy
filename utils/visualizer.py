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
            cv2.putText(
                image,
                f"{joint}: {angle:.1f} deg",  # 使用 'deg' 替代 '°' 符号
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y_pos += 30
