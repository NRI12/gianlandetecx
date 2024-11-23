import numpy as np
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 255, 0)  # Màu xanh lá cây
thickness = 2

class Camera:
    def __init__(self, video_source=0):
        # Sử dụng 0 cho camera mặc định hoặc đường dẫn tệp video
        self.video = cv2.VideoCapture(video_source)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        return frame
