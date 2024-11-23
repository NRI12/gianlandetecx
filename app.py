import cv2
import numpy as np
from modules.SCRFD import SCRFD
def visualize(image, boxes, lmarks, scores, fps=0):
    for i in range(len(boxes)):
        print(boxes[i])
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

    
# load scrfd face detector model
onnxmodel = 'models/scrfd_500m_kps.onnx'
confThreshold = 0.5
nmsThreshold = 0.5
mynet = SCRFD(onnxmodel)

deviceId = 0# select camera
cap = cv2.VideoCapture(deviceId)

tm = cv2.TickMeter()

while cv2.waitKey(1) < 0:
    
    hasFrame, frame = cap.read()
    if not hasFrame:
        print('No frames captured!')
        break
    frame = cv2.flip(frame, 1)

    # Inference
    tm.start()# for calculating FPS
    bboxes, lmarks, scores = mynet.detect(frame)# face detection
    tm.stop()
    
    # process if at least one face detected
    if bboxes.shape[0] > 0 or lmarks.shape[0] > 0:
        
        # Draw results on the input image
        frame = visualize(frame, bboxes, lmarks, scores, fps=tm.getFPS())
        
        # Check if all coordinates of the highest score face in the frame

            
        roll, yaw, pitch = find_pose(lmarks[0])
        
        # visualize pose
        lmarks = lmarks.astype(int)
        start_point = (lmarks[0][2][0], lmarks[0][2][1])
        end_point = (lmarks[0][2][0]-int(yaw), lmarks[0][2][1]-int(pitch))
        
        cv2.arrowedLine(frame, start_point, end_point, (255,0,0), 2)
        bn = "\n"
        cv2.putText(frame, f"roll: {int(roll)} -- yaw: {int(yaw)} -- pitch: {int(pitch)}", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
    
    # Visualize results in a new Window
    cv2.imshow('Face Pose', frame)
    #cv2.waitKey(0)

    tm.reset()


cv2.destroyAllWindows()
cap.release()
