import cv2
import mediapipe as mp
import numpy as np
import time
import math

# -----------------------------
# MediaPipe
# -----------------------------
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# -----------------------------
# MODE (IMPORTANT FIX)
# -----------------------------
mode = "B"

# -----------------------------
# RGB cycle
# -----------------------------
def rgb(i=0):
    t = time.time() * 2 + i * 0.1
    r = int((math.sin(t) + 1) * 127)
    g = int((math.sin(t + 2) + 1) * 127)
    b = int((math.sin(t + 4) + 1) * 127)
    return (b, g, r)

# -----------------------------
# FACE OUTLINE (FIXED PROPER MASK)
# jaw + face oval = real silhouette
# -----------------------------
FACE_OUTLINE = list(set([
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21
]))

# -----------------------------
# DRAW FACE (neon outline mask)
# -----------------------------
def draw_face(canvas, lm, w, h):
    pts = []

    for i, idx in enumerate(FACE_OUTLINE):
        p = lm[idx]
        x, y = int(p.x * w), int(p.y * h)
        col = rgb(i)

        cv2.circle(canvas, (x, y), 4, col, -1)
        cv2.circle(canvas, (x, y), 7, tuple(c // 2 for c in col), 1)

        pts.append((x, y))

    # connect outline (IMPORTANT FIX)
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], rgb(i), 2)

# -----------------------------
# DRAW HANDS (unchanged style)
# -----------------------------
def draw_hand(canvas, lm, w, h):
    for i, conn in enumerate(mp_hands.HAND_CONNECTIONS):
        s, e = conn
        x1, y1 = int(lm[s].x * w), int(lm[s].y * h)
        x2, y2 = int(lm[e].x * w), int(lm[e].y * h)

        cv2.line(canvas, (x1, y1), (x2, y2), rgb(i), 2)

    for i in range(21):
        x, y = int(lm[i].x * w), int(lm[i].y * h)
        cv2.circle(canvas, (x, y), 4, rgb(i), -1)

# -----------------------------
# LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_res = face_mesh.process(rgb_frame)
    hand_res = hands_detector.process(rgb_frame)

    # -----------------------------
    # HANDS
    # -----------------------------
    if mode in ["H", "B"]:
        if hand_res.multi_hand_landmarks:
            for hand in hand_res.multi_hand_landmarks:
                draw_hand(canvas, hand.landmark, w, h)

    # -----------------------------
    # FACE (FULL OUTLINE FIXED)
    # -----------------------------
    if mode in ["F", "B"]:
        if face_res.multi_face_landmarks:
            for face in face_res.multi_face_landmarks:
                draw_face(canvas, face.landmark, w, h)

    # -----------------------------
    # INPUT FIX (reliable switching)
    # -----------------------------
    key = cv2.waitKey(1)
    if key != -1:
        k = chr(key & 255).lower()

        if k == "h":
            mode = "H"
            print("MODE → HANDS")
        elif k == "f":
            mode = "F"
            print("MODE → FACE")
        elif k == "b":
            mode = "B"
            print("MODE → BOTH")
        elif k == "q":
            break

    cv2.putText(canvas, f"MODE: {mode}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                rgb(1), 1)

    cv2.imshow("RGB NEON AR SYSTEM", canvas)

cap.release()
cv2.destroyAllWindows()