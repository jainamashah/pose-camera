import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2)

# ----------------------
# Parameters
# ----------------------
HEIGHT_THRESH = 0.06     # how much higher than standing
VEL_THRESH = -0.002
LANDING_THRESH = 0.015

hip_buffer = deque(maxlen=30)
vel_buffer = deque(maxlen=10)

jump_count = 0
in_jump = False
peak_taken = False

SAVE_DIR = "jump_photos"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        lh = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

        if lh.visibility > 0.6 and rh.visibility > 0.6:
            hip_y = (lh.y + rh.y) / 2
            hip_buffer.append(hip_y)

            if len(hip_buffer) > 1:
                vel = hip_buffer[-1] - hip_buffer[-2]
                vel_buffer.append(vel)

                ground_level = np.median(hip_buffer)

                # -------- JUMP START --------
                if not in_jump:
                    if hip_y < ground_level - HEIGHT_THRESH and vel < VEL_THRESH:
                        in_jump = True
                        peak_taken = False

                # -------- PEAK --------
                if in_jump and not peak_taken:
                    if vel_buffer[-2] < 0 and vel >= 0:
                        jump_count += 1
                        cv2.imwrite(f"{SAVE_DIR}/jump_{jump_count}.jpg", frame)
                        print(f"📸 Jump {jump_count} peak captured")
                        peak_taken = True

                # -------- LANDING --------
                if in_jump:
                    if abs(hip_y - ground_level) < LANDING_THRESH and vel > 0:
                        in_jump = False

                # Debug overlay
                cv2.putText(frame, f"hip: {hip_y:.3f}", (20,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"vel: {vel:.4f}", (20,110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(frame, f"jumping: {in_jump}", (20,140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        mp.solutions.drawing_utils.draw_landmarks(
            frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    cv2.putText(frame, f"Jumps: {jump_count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Reliable Jump Detector", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
