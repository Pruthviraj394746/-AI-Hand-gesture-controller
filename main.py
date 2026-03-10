import os
# Workaround for multiple OpenMP copies (often caused by torch/cv2)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import mediapipe as mp
import cv2
import time
import math
import sys
import threading
import traceback
import numpy as np

from core.hand_tracker import HandTracker
from core.feature_extractor import extract_features, fingers_up, calculate_distance
from core.gesture_recognizer import GestureRecognizer
from core.system_controller import SystemController
from core.smoothing import CursorSmoother
from ui.overlay_ui import OverlayUI
from ui.main_window import run_gui
from utils.logger import setup_logger

class HandGestureApp:
    def __init__(self):
        self.running = False
        self.camera_thread = None
        self.alpha = 0.5
        self.logger = setup_logger("HandGestureApp")
        self.controller = None 
        
    def start_app(self, alpha=0.5):
        self.alpha = alpha
        if not self.running:
            self.running = True
            self.camera_thread = threading.Thread(target=self.run_camera_loop, daemon=True)
            self.camera_thread.start()
            self.logger.info(f"App started with sensitivity alpha={self.alpha}")

    def stop_app(self):
        self.running = False
        self.logger.info("App stopped")

    def run_camera_loop(self):
        cap = cv2.VideoCapture(0)
        
        # Resolutions
        cam_w, cam_h = 640, 480
        cap.set(3, cam_w)
        cap.set(4, cam_h)
        
        # Modules setup
        tracker = HandTracker(max_hands=1)
        recognizer = GestureRecognizer(model_path="models/keypoint_classifier.pth", num_classes=5)
        
        # Isolated SystemController to avoid COM conflict if possible
        try:
            self.controller = SystemController(screen_w=cam_w, screen_h=cam_h)
        except Exception as e:
            self.logger.error(f"Failed to initialize SystemController: {e}")
            self.controller = None
            
        smoother = CursorSmoother(alpha=self.alpha)
        overlay = OverlayUI(frame_w=cam_w, frame_h=cam_h)
        
        pTime = 0
        current_mode = "Mouse"
        last_gesture_time = 0
        
        while self.running:
            try:
                success, img = cap.read()
                if not success:
                    self.logger.error("Failed to read from camera.")
                    break
                    
                img = cv2.flip(img, 1) # Mirror
                
                # FPS Calculation
                cTime = time.time()
                fps = 1 / (cTime - pTime) if pTime != 0 else 0
                pTime = cTime
                
                img = tracker.find_hands(img, draw=True)
                lm_list = tracker.find_position(img, draw=False)
                
                gesture_name = "None"
                conf = 0.0
                
                if lm_list:
                    # 1. Extract finger heuristics
                    fingers = fingers_up(lm_list)
                    x1, y1 = lm_list[8][1], lm_list[8][2]
                    
                    # 2. Gesture Recognition (Heuristic)
                    gesture_name, conf = recognizer.predict(None, lm_list, fingers)
                    
                    # Check for UI interaction
                    hit_mode = overlay.check_button_hit(x1, y1)
                    if hit_mode and fingers == [0, 1, 0, 0, 0]: # Pointing
                        current_mode = hit_mode
                    
                    # 3. Action Mapping based on Mode
                    dist = calculate_distance((lm_list[4][1], lm_list[4][2]), (lm_list[8][1], lm_list[8][2]))
                    is_click = (dist < 40 and fingers[1] == 1)
                    
                    if is_click:
                        cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
                    
                    # Perform action
                    if self.controller:
                        try:
                            self.controller.perform_action(current_mode, gesture_name, lm_list, fingers, is_click)
                        except Exception as e:
                            self.logger.error(f"Controller error: {e}")
                    else:
                        if gesture_name != "None":
                            self.logger.debug(f"DEBUG: Mode={current_mode}, Gesture={gesture_name}, Click={is_click}")
                
                # 4. Draw Overlay UI
                img = overlay.draw_ui(img, fps, current_mode, gesture_name, conf)
                
                cv2.imshow("Hand Gesture Controller", img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
            except Exception as e:
                self.logger.error(f"Error in camera loop: {e}")
                self.logger.error(traceback.format_exc())
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("DEBUG: Application starting...")
    app = HandGestureApp()
    print("DEBUG: HandGestureApp initialized.")
    
    try:
        print("DEBUG: Running GUI...")
        run_gui(app.start_app, app.stop_app)
        print("DEBUG: GUI closed.")
    except Exception as e:
        print(f"DEBUG Error: {e}")
        app.stop_app()
