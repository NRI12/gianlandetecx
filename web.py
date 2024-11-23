from importlib import import_module
import os
from flask import Flask, render_template, Response, request, redirect, url_for
from video_play import Camera
import time
import requests
import cv2
import threading
import cv2
import numpy as np
from modules.SCRFD import SCRFD
import pygame
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
pygame.init()
source = 'rtsp://192.168.1.14:554/onvif1'  # Mặc định là sử dụng camera
pygame.mixer.init()
violation_counts = {
    'left': 0,
    'right': 0,
    'up': 0,
    'down': 0,
    'multiple_people_count': 0
}
user_id = 0
violated = {
    'left': False,
    'right': False,
    'up': False,
    'down': False,
    'multiple_people_count': False,
}
roomParticipantId_temp = None

TELEGRAM_BOT_TOKEN = "www"
TELEGRAM_CHAT_ID = "www"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
TELEGRAM_PHOTO_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
def visualize(image, boxes, lmarks, scores, fps=0):
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax,score = boxes[i].astype('int')
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
        for j in range(5):
            cv2.circle(image, (int(lmarks[i, j, 0]), int(lmarks[i, j, 1])), 1, (0,255,0), thickness=-1)
    return image        

def are_coordinates_in_frame(frame, box, pts):
    
    height, width = frame.shape[:2]
    
    if np.any(box <= 0) or np.any(box >= height) or np.any(box >= width):
        return False
    if np.any(pts <= 0) or np.any(pts >= height) or np.any(pts >= width):
        return False
    
    return True

def find_pose(points):
    LMx = points[:,0]#points[0:5]# horizontal coordinates of landmarks
    LMy = points[:,1]#[5:10]# vertical coordinates of landmarks
    
    dPx_eyes = max((LMx[1] - LMx[0]), 1)
    dPy_eyes = (LMy[1] - LMy[0])
    angle = np.arctan(dPy_eyes / dPx_eyes) # angle for rotation based on slope
    
    alpha = np.cos(angle)
    beta = np.sin(angle)
    
    # rotated landmarks
    LMxr = (alpha * LMx + beta * LMy + (1 - alpha) * LMx[2] / 2 - beta * LMy[2] / 2) 
    LMyr = (-beta * LMx + alpha * LMy + beta * LMx[2] / 2 + (1 - alpha) * LMy[2] / 2)
    
    # average distance between eyes and mouth
    dXtot = (LMxr[1] - LMxr[0] + LMxr[4] - LMxr[3]) / 2
    dYtot = (LMyr[3] - LMyr[0] + LMyr[4] - LMyr[1]) / 2
    
    # average distance between nose and eyes
    dXnose = (LMxr[1] - LMxr[2] + LMxr[4] - LMxr[2]) / 2
    dYnose = (LMyr[3] - LMyr[2] + LMyr[4] - LMyr[2]) / 2
    
    # relative rotation 0 degree is frontal 90 degree is profile
    Xfrontal = (-90+90 / 0.5 * dXnose / dXtot) if dXtot != 0 else 0
    Yfrontal = (-90+90 / 0.5 * dYnose / dYtot) if dYtot != 0 else 0

    return angle * 180 / np.pi, Xfrontal, Yfrontal
def play_audio(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    pygame.time.set_timer(pygame.USEREVENT, 3000)

onnxmodel = 'models/scrfd_500m_kps.onnx'
confThreshold = 0.5
nmsThreshold = 0.5
mynet = SCRFD(onnxmodel)
def report_violation(room_participant_id, violation_data):
    url = 'http://localhost:8000/api/violations'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'room_participant_id': room_participant_id,
        'left': violation_data['left'],
        'right': violation_data['right'],
        'up': violation_data['up'],
        'down': violation_data['down'],
        'multiple_people_count': violation_data['multiple_people_count']
    }
    print(f"Violation reported: 12413123333333333333 ")

    response = requests.post(url, json=payload, headers=headers)
    return response

def handle_violations(frame, violation_counts, violated):
    global roomParticipantId_temp 
    violation_data = {
        'left': violation_counts['left'],
        'right': violation_counts['right'],
        'up': violation_counts['up'],
        'down': violation_counts['down'],
        'multiple_people_count': violation_counts['multiple_people_count']
    }
    
    # Only call the API if there's a violation
    if any(violated.values()):
        # Report violation to Laravel API
        response = report_violation(roomParticipantId_temp, violation_data)
        print(f"Violation reported: {response.status_code} , {roomParticipantId_temp} , {response.json()}")

    # Inside your main frame loop, add:
    if any(violated.values()):
        handle_violations(frame, violation_counts, violated)
@app.route("/")
def index():
    """Trang chủ phát video."""
    return render_template("index.html")


@app.route("/set_source", methods=["POST"])
def set_source():
    global source
    source = request.form['source']
    if source:
        print(f"Source set to: {source}")
        return 'Source updated successfully', 200
    else:
        return 'Failed to set source', 400


@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        global source
        source = 'uploaded_video'
        app.config['UPLOADED_VIDEO_PATH'] = file_path
        return redirect(url_for('index'))


def gen(camera):
    frame_count = 0
    start_time = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX
    tm = cv2.TickMeter()
    font_scale = 1       # Kích thước font chữ
    color = (0, 255, 0)  # Màu chữ (xanh lá cây)
    thickness = 2
    count_smoke = 0
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        frame = cv2.flip(frame, 1)
        tm.start()# for calculating FPS
        bboxes, lmarks, scores = mynet.detect(frame)# face detection
        tm.stop()
        
        if bboxes.shape[0] > 0 or lmarks.shape[0] > 0:
            frame = visualize(frame, bboxes, lmarks, scores, fps=tm.getFPS())
            
            # Check if all coordinates of the highest score face in the frame
            position = 'normal'

            roll, yaw, pitch = find_pose(lmarks[0])
            if bboxes.shape[0] >= 2:  # Kiểm tra có từ 2 khuôn mặt trở lên
                    if not violated['multiple_people_count']:  # Nếu chưa ghi nhận vi phạm trước đó
                        violation_counts['multiple_people_count'] += 1  # Ghi nhận vi phạm
                        violated['multiple_people_count'] = True  # Đánh dấu vi phạm đã xảy ra
                        cv2.putText(frame, f"Multiple people detected!", (20, 220), font, font_scale, (0, 0, 255), thickness)
                        handle_violations(frame, violation_counts, violated)  # Report violation

                    else:
                        # Reset trạng thái khi số lượng người trở lại bình thường (< 2 người)
                        violated['multiple_people_count'] = False

            if yaw > 40:  # Left violation
                 if not violated['left']:  # Chỉ tính nếu chưa vi phạm trước đó
                    violation_counts['left'] += 1
                    violated['left'] = True  # Đánh dấu đã vi phạm
                    threading.Thread(target=play_audio, args=("amthanh/trai.mp3",)).start()
                    handle_violations(frame, violation_counts, violated)  # Report violation

            elif yaw < -40:  # Right violation
                if not violated['right']:  # Chỉ tính nếu chưa vi phạm trước đó
                    violation_counts['right'] += 1
                    violated['right'] = True  # Đánh dấu đã vi phạm
                    threading.Thread(target=play_audio, args=("amthanh/phai.mp3",)).start()
                    handle_violations(frame, violation_counts, violated)  # Report violation

            else:  # Quay về trạng thái bình thường
                violated['left'] = False
                violated['right'] = False
            if pitch > 25:  # Upward violation
                if not violated['up']:  # Chỉ tính nếu chưa vi phạm trước đó
                    violation_counts['up'] += 1
                    violated['up'] = True  # Đánh dấu đã vi phạm
                    threading.Thread(target=play_audio, args=("amthanh/tren.mp3",)).start()
            elif  pitch < -25:  # Downward violation
                if not violated['down']:  # Chỉ tính nếu chưa vi phạm trước đó
                    violation_counts['down'] += 1
                    violated['down'] = True  # Đánh dấu đã vi phạm
                    threading.Thread(target=play_audio, args=("amthanh/xuong.mp3",)).start()
            else:  # Quay về trạng thái bình thường
                    violated['up'] = False
                    violated['down'] = False
            # Log the current violation counts
            print(f"Violations - Left: {violation_counts['left']}, Right: {violation_counts['right']}, Up: {violation_counts['up']}, multiple_people_count: {violation_counts['multiple_people_count']}")

            lmarks = lmarks.astype(int)
            start_point = (lmarks[0][2][0], lmarks[0][2][1])
            end_point = (lmarks[0][2][0] - int(yaw), lmarks[0][2][1] - int(pitch))

            cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 2)
            bn = "\n"
            cv2.putText(frame, f"{position}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        else:
            threading.Thread(target=play_audio,args=("amthanh/khongphathienkhuonmat.mp3",)).start()
            cv2.putText(frame, f"khong phat hien khuon mat", 
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
            

        cv2.putText(frame, f"Violations Left: {violation_counts['left']}", (20, 100), font, font_scale, color, thickness)
        cv2.putText(frame, f"Right: {violation_counts['right']}", (20, 130), font, font_scale, color, thickness)
        cv2.putText(frame, f"Up: {violation_counts['up']}", (20, 160), font, font_scale, color, thickness)
        cv2.putText(frame, f"Down: {violation_counts['down']}", (20, 190), font, font_scale, color, thickness)
        # Tính toán FPS
        frame_count += 1
        end_time = time.time()
        elapsed_time = end_time - start_time

        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        else:
            fps = 0

        # Làm tròn FPS về số nguyên
        fps_int = int(round(fps))

        # Vẽ FPS lên khung hình
        cv2.putText(frame, f"FPS: {fps_int}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Mã hóa khung hình ở định dạng JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # Đặt lại bộ đếm và thời gian bắt đầu sau mỗi giây
        if elapsed_time > 1:
            frame_count = 0
            start_time = time.time()


@app.route("/video_feed")
def video_feed():
    """Route phát video. Truyền tham số user_id và exam_id qua query string."""
    global roomParticipantId_temp
    roomParticipantId_temp = request.args.get('roomParticipantId')  # Lấy tham số user_id từ URL
    user_id = request.args.get('roomParticipantId')  # Lấy tham số user_id từ URL
    
    # In ra các tham số để kiểm tra
    print(f"User ID: {user_id} ")

    global source
    if source == 'video':
        # Đường dẫn tới video của bạn
        video_path = "C:/Users/84986/Desktop/hotrogiaotiepcamdiec/testVD.mp4"
        camera = Camera()
        camera.video = cv2.VideoCapture(video_path)
    elif source == 'uploaded_video':
        video_path = app.config.get('UPLOADED_VIDEO_PATH')
        camera = Camera()
        camera.video = cv2.VideoCapture(video_path)
    else:
        camera = Camera()
    
    return Response(gen(camera), mimetype="multipart/x-mixed-replace; boundary=frame")
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, threaded=True)
