# virtual_painter.py
import cv2
import numpy as np
import mediapipe as mp
import time
from datetime import datetime

# ---------- Helpers ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def fingers_up(hand_landmarks):
    """
    Returns booleans for index, middle, ring, pinky fingers (True if finger is up).
    Uses landmark-tip.y < landmark-pip.y convention (works for non-thumb fingers).
    """
    tips = [8, 12, 16, 20]
    states = []
    for tip in tips:
        states.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y)
    return states  # [index, middle, ring, pinky]

# ---------- Palette config ----------
PALETTE = [
    ("Red",    (0, 0, 255)),
    ("Green",  (0, 255, 0)),
    ("Blue",   (255, 0, 0)),
    ("Yellow", (0, 255, 255)),
    ("Eraser", (0, 0, 0)),   # selecting Eraser will switch to eraser mode
    ("Clear",  (200, 200, 200))
]

RECT_W = 110
RECT_H = 80
RECT_Y = 10
RECT_SPACING = 10
START_X = 10

# ---------- Settings ----------
BRUSH_THICKNESS = 12
ERASER_THICKNESS = 50
SELECTION_COOLDOWN = 0.4  # seconds between palette selections to avoid flicker

# ---------- Camera ----------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

canvas = None
current_color = (0, 0, 255)  # default red
palette_eraser = False
prev_x, prev_y = 0, 0
last_selection_time = 0

# Precompute palette rectangles
palette_rects = []
x = START_X
for name, color in PALETTE:
    palette_rects.append((x, RECT_Y, x + RECT_W, RECT_Y + RECT_H, name, color))
    x += RECT_W + RECT_SPACING

# ---------- Main loop ----------
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw palette on the visible frame (UI only)
        for (x1, y1, x2, y2, name, color) in palette_rects:
            # draw Eraser as white box for clarity
            fill_color = (255, 255, 255) if name == "Eraser" or name == "Clear" else color
            cv2.rectangle(frame, (x1, y1), (x2, y2), fill_color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), 2)
            cv2.putText(frame, name, (x1 + 8, y2 - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # fingertip coordinates
            x_index = int(hand_landmarks.landmark[8].x * w)
            y_index = int(hand_landmarks.landmark[8].y * h)

            # check palette selection (if fingertip is in top area)
            now = time.time()
            for (x1, y1, x2, y2, name, color) in palette_rects:
                if x1 < x_index < x2 and y1 < y_index < y2 and (now - last_selection_time) > SELECTION_COOLDOWN:
                    last_selection_time = now
                    if name == "Clear":
                        canvas[:] = 0
                    elif name == "Eraser":
                        palette_eraser = True
                        current_color = (0, 0, 0)
                    else:
                        palette_eraser = False
                        current_color = color

            # finger-up states
            f_states = fingers_up(hand_landmarks)  # [index, middle, ring, pinky]
            index_up = f_states[0]
            middle_up = f_states[1]

            # eraser by gesture (index + middle up) OR palette eraser selected
            gesture_eraser = index_up and middle_up
            eraser_mode = gesture_eraser or palette_eraser

            # Drawing: index up only
            if index_up and not middle_up:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x_index, y_index

                if eraser_mode:
                    cv2.line(canvas, (prev_x, prev_y), (x_index, y_index), (0, 0, 0), ERASER_THICKNESS)
                else:
                    cv2.line(canvas, (prev_x, prev_y), (x_index, y_index), current_color, BRUSH_THICKNESS)
                prev_x, prev_y = x_index, y_index
            else:
                prev_x, prev_y = 0, 0

        # Overlay the canvas on the frame using mask so black (0,0,0) is transparent
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        drawing_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        combined = cv2.add(frame_bg, drawing_fg)

        # Instruction text
        cv2.putText(combined, "Index = draw | Index+Middle = erase | Touch top boxes to select",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Virtual Painter", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # Esc or q
            break
        elif key == ord('c'):
            canvas[:] = 0
        elif key == ord('s'):
            # Save painting on white bg (without UI palette)
            bg = np.ones_like(frame, dtype=np.uint8) * 255
            bg_bg = cv2.bitwise_and(bg, bg, mask=mask_inv)
            painting_on_white = cv2.bitwise_and(canvas, canvas, mask=mask)
            saved = cv2.add(bg_bg, painting_on_white)

            fname = f"painting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fname, saved)
            print("Saved:", fname)

cap.release()
cv2.destroyAllWindows()
