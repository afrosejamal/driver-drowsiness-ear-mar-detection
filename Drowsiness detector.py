import os
os.environ["GLOG_minloglevel"] = "2"

import cv2
import dlib
import math
import time

# ======================
# HELPER FUNCTIONS
# ======================
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_EAR(eye):
    # Eye landmarks: [p1, p2, p3, p4, p5, p6]
    v1 = euclidean_distance(eye[1], eye[5])
    v2 = euclidean_distance(eye[2], eye[4])
    h = euclidean_distance(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h)

def calculate_MAR(mouth):
    # Mouth landmarks: [p61 ... p68] (inner mouth points)
    vertical = euclidean_distance(mouth[13], mouth[19])   # upper-lip to lower-lip
    horizontal = euclidean_distance(mouth[12], mouth[16])  # mouth corners
    return vertical / horizontal

# ======================
# THRESHOLDS
# ======================
EAR_THRESHOLD = 0.25     # Below this = eyes considered closed
MAR_THRESHOLD = 0.6      # Above this = mouth considered open (yawn)
DROWSY_FRAMES = 20       # Consecutive frames below EAR threshold before alerting
YAWN_FRAMES = 15         # Consecutive frames above MAR threshold before alerting

eye_counter = 0
mouth_counter = 0

# ======================
# VIDEO SOURCE
# ======================
# Use 0 for your default webcam, or replace with a path to a video file
# e.g. VIDEO_SOURCE = "input_videos/test1.mp4"
VIDEO_SOURCE = 0

print("Connecting to video source...")
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("ERROR: Could not open video source")
    exit()

print("Video source connected")

# ======================
# DLIB FACE DETECTOR & PREDICTOR
# ======================
# Download shape_predictor_68_face_landmarks.dat from:
# https://github.com/davisking/dlib-models
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(predictor_path):
    raise FileNotFoundError(
        f"'{predictor_path}' not found. Download it from "
        "
    )

predictor = dlib.shape_predictor(predictor_path)
print("dlib face detector & predictor initialized")

# ======================
# FACIAL LANDMARK INDICES (standard 68-point dlib model)
# ======================
LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
MOUTH_IDX = list(range(48, 68))

# ======================
# MAIN LOOP
# ======================
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video / frame not received")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = [landmarks[i] for i in LEFT_EYE_IDX]
        right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]
        mouth = [landmarks[i] for i in MOUTH_IDX]

        EAR = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2
        MAR = calculate_MAR(mouth)

        # Draw eye landmarks
        for p in left_eye + right_eye:
            cv2.circle(frame, p, 2, (0, 255, 255), -1)

        # Draw mouth landmarks
        for p in mouth:
            cv2.circle(frame, p, 2, (255, 0, 255), -1)

        # DROWSINESS check (sustained eye closure)
        if EAR < EAR_THRESHOLD:
            eye_counter += 1
            if eye_counter >= DROWSY_FRAMES:
                cv2.putText(frame, "DROWSY!", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            eye_counter = 0

        # FATIGUE/YAWN check (sustained mouth opening)
        if MAR > MAR_THRESHOLD:
            mouth_counter += 1
            if mouth_counter >= YAWN_FRAMES:
                cv2.putText(frame, "FATIGUE (YAWNING)", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 3)
        else:
            mouth_counter = 0

        # Display live EAR/MAR values
        cv2.putText(frame, f"EAR: {EAR:.2f}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {MAR:.2f}", (30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Driver Drowsiness & Fatigue Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
