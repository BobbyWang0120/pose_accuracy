import cv2
import mediapipe as mp
from utils.angle_calculator import AngleCalculator
from utils.visualizer import PoseVisualizer


class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.visualizer = PoseVisualizer()
        self.angle_calculator = AngleCalculator()

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # 处理图像
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                # 绘制骨骼点和连接线
                self.visualizer.draw_landmarks(image, results)

                # 计算并显示角度
                angles = self.angle_calculator.get_fitness_angles(results.pose_landmarks.landmark)
                self.visualizer.draw_angles(image, angles)

            cv2.imshow("Fitness Pose Analysis", image)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = PoseDetector()
    detector.run()
