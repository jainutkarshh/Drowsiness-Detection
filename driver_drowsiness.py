import cv2
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
from imutils import face_utils
import simpleaudio as sa
import threading
import time
from datetime import datetime
import os
import math

if not os.path.exists("logs"):
    os.makedirs("logs")

log_filename = f"logs/drowsiness_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(log_filename, "w") as f:
    f.write("timestamp,eye_ratio,head_angle,status,duration\n")

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

        # Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")
except RuntimeError as e:
    print(f"Error: {e}")
    cap.release()
    exit()

sleep = 0
drowsy = 0
active = 0
status = "Active :)"
color = (0, 255, 0)
last_alarm_time = 0     
alarm_interval = 10     # seconds between alarms
state_start_time = time.time()  
blink_count = 0
blink_start = time.time()
blinks_per_minute = 0


eye_ratio_history = []
head_angle_history = []

# Calibration values
eye_open_ratio = 0.25 
eye_drowsy_ratio = 0.21  
drowsiness_score = 0.0

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def get_eye_ratio(eye_points):
    # Calculate the eye aspect ratio (EAR)
    # EAR = (d1 + d2) / (2 * d3)
    # where d1 and d2 are the vertical distances and d3 is the horizontal distance
    a = compute(eye_points[1], eye_points[5])
    b = compute(eye_points[2], eye_points[4])
    c = compute(eye_points[0], eye_points[3])
    return (a + b) / (2.0 * c)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

   
    if ratio > eye_open_ratio:
        return 2
    elif eye_drowsy_ratio < ratio <= eye_open_ratio:
        return 1
    else:
        return 0

def get_head_pose(landmarks):
    
    forehead = landmarks[27]
    chin = landmarks[8]
    
   
    angle = math.degrees(math.atan2(chin[1] - forehead[1], chin[0] - forehead[0]))
    return angle

def play_alarm(level=2):
   
    alarm_files = ["gentle_alarm.wav", "medium_alarm.wav", "alarm.wav"]
    filename = alarm_files[min(level, 2)]
    
    try:
        wave_obj = sa.WaveObject.from_wave_file(filename)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except:
        
        try:
            wave_obj = sa.WaveObject.from_wave_file("alarm.wav")
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as e:
            print(f"Error playing alarm: {e}")

def log_drowsiness_data(timestamp, eye_ratio, head_angle, status, duration):
    with open(log_filename, "a") as f:
        f.write(f"{timestamp},{eye_ratio:.4f},{head_angle:.2f},{status},{duration:.2f}\n")

def calibrate_eye_ratio():
    print("Calibration: Please look at the camera with eyes fully open for 5 seconds")
    ratios = []
    for _ in range(50): 
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            left_ratio = get_eye_ratio(left_eye)
            right_ratio = get_eye_ratio(right_eye)
            
            ratios.append((left_ratio + right_ratio) / 2)
            
        cv2.putText(frame, "Calibrating: Keep eyes open", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(100) == 27:      # ESC key to exit calibration
            break
            
    if ratios:
        open_ratio = np.mean(ratios)
        drowsy_ratio = open_ratio * 0.8
        closed_ratio = open_ratio * 0.70
        print(f"Calibration complete: Open: {open_ratio:.4f}, Drowsy: {drowsy_ratio:.4f}, Closed: {closed_ratio:.4f}")
        return open_ratio, drowsy_ratio
    else:
        print("Calibration failed: No face detected")
        return 0.25, 0.21  

def create_dashboard(frame, eye_ratio, head_angle, blink_rate, drowsiness_score, duration):
    
    dashboard = np.zeros((150, frame.shape[1], 3), dtype=np.uint8)
    
    
    cv2.putText(dashboard, f"Blink Rate: {blink_rate:.1f} blinks/min", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(dashboard, f"Eye Ratio: {eye_ratio:.2f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(dashboard, f"Head Angle: {head_angle:.1f}°", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(dashboard, f"Status Duration: {duration:.1f}s", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    
    cv2.putText(dashboard, status, (frame.shape[1]-200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
   
    meter_width = int(drowsiness_score * 200)
    cv2.rectangle(dashboard, (frame.shape[1]-200, 70), (frame.shape[1]-200+meter_width, 90), 
                  (0, 0, 255), -1)
    cv2.rectangle(dashboard, (frame.shape[1]-200, 70), (frame.shape[1]-200+200, 90), 
                  (255, 255, 255), 1)
    
    
    combined = np.vstack([frame, dashboard])
    return combined

print("Starting eye calibration...")
eye_open_ratio, eye_drowsy_ratio = calibrate_eye_ratio()

frame_skip = 0  
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame_count += 1
    if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    face_frame = frame.copy()
    
   
    current_eye_ratio = 0.3 
    current_head_angle = 0
    
    
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        left_ratio = get_eye_ratio(left_eye)
        right_ratio = get_eye_ratio(right_eye)
        
        current_eye_ratio = (left_ratio + right_ratio) / 2
      
        eye_ratio_history.append(current_eye_ratio)
        if len(eye_ratio_history) > 10: 
            eye_ratio_history.pop(0)
        
       
        smoothed_eye_ratio = np.mean(eye_ratio_history)
        
      
        current_head_angle = get_head_pose(landmarks)
        head_angle_history.append(current_head_angle)
        if len(head_angle_history) > 10:
            head_angle_history.pop(0)
        
        smoothed_head_angle = np.mean(head_angle_history)
        
        left_blink = blinked(landmarks[36], landmarks[37], 
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], 
                             landmarks[44], landmarks[47], landmarks[46], landmarks[45])
       
        if (left_blink == 0 or right_blink == 0) and (left_blink == 2 or right_blink == 2):
            blink_count += 1
        
       
        elapsed = time.time() - blink_start
        if elapsed >= 60:
            blinks_per_minute = blink_count / (elapsed / 60)
            blink_count = 0
            blink_start = time.time()
        
        current_time = time.time()
        
        eye_factor = max(0, min(1, (eye_open_ratio - smoothed_eye_ratio) / (eye_open_ratio - eye_drowsy_ratio)))
        head_factor = abs(smoothed_head_angle) / 30.0  # Normalize head angle (±30 degrees considered max)
        
        # Combined drowsiness score (70% eye, 30% head)
        drowsiness_score = 0.7 * eye_factor + 0.3 * min(1.0, head_factor)
        
        if left_blink == 0 or right_blink == 0:
            if status != "SLEEPING !!!":
                state_start_time = current_time
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (0, 0, 255)  
                if current_time - state_start_time > 2:
                    if current_time - last_alarm_time > alarm_interval:
                        threading.Thread(target=play_alarm, args=(2,)).start()
                        last_alarm_time = current_time
                        
        elif left_blink == 1 or right_blink == 1 or drowsiness_score > 0.6:
            if status != "Drowsy !":
                state_start_time = current_time
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 140, 255)  # Orange
                if current_time - state_start_time > 2:
                    if current_time - last_alarm_time > alarm_interval:
                        threading.Thread(target=play_alarm, args=(1,)).start()
                        last_alarm_time = current_time
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)  # Green
        
       
        state_duration = current_time - state_start_time
        
        # Log data every 2 seconds
        if int(current_time) % 2 == 0:
            log_drowsiness_data(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                smoothed_eye_ratio,
                smoothed_head_angle,
                status,
                state_duration
            )
        
        # Display facial landmarks
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
            
      
        hull_left = cv2.convexHull(left_eye)
        hull_right = cv2.convexHull(right_eye)
        cv2.drawContours(face_frame, [hull_left], -1, (0, 255, 0), 1)
        cv2.drawContours(face_frame, [hull_right], -1, (0, 255, 0), 1)
        
   
    cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
 
    dashboard_frame = create_dashboard(
        frame, 
        current_eye_ratio, 
        current_head_angle, 
        blinks_per_minute, 
        drowsiness_score,
        current_time - state_start_time
    )

    # Show results
    cv2.imshow("Driver Monitoring", dashboard_frame)
    cv2.imshow("Facial Landmarks", face_frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break
    elif key == ord('r'):  # Recalibrate
        print("Recalibrating...")
        eye_open_ratio, eye_drowsy_ratio = calibrate_eye_ratio()

cap.release()
cv2.destroyAllWindows()
print(f"Session data logged to {log_filename}")