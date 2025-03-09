import sys
import cv2
import numpy as np
import math
import time
import os
import pyautogui
import screen_brightness_control as sbc
import tensorflow as tf
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from pynput import keyboard

import handtrackingmodule as htm

# Platform-specific imports
import platform
is_mac = platform.system() == "Darwin"
is_windows = platform.system() == "Windows"
if is_windows:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER

# Optional GPU check for TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and enabled")
else:
    print("GPU is not available, using CPU")

# --- Helper Drawing Functions ---
def draw_text_with_bg(frame, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7,
                      text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=2):
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x-5, y-h-5), (x+w+5, y+5), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

def draw_slider(frame, pos, size, percentage, slider_color, bg_color=(50,50,50)):
    x, y = pos
    width, height = size
    cv2.rectangle(frame, (x, y), (x+width, y+height), bg_color, -1)
    fill_height = int(height * (percentage / 100.0))
    cv2.rectangle(frame, (x, y+height-fill_height), (x+width, y+height), slider_color, -1)
    cv2.rectangle(frame, (x, y), (x+width, y+height), (255,255,255), 2)

# --- Platform-Specific Functions ---
def set_volume_mac(level):
    os.system(f"osascript -e 'set volume output volume {int(level)}'")

def initialize_windows_audio():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return interface.QueryInterface(IAudioEndpointVolume)

# ------------------------------
# Video Processing Thread
# ------------------------------
class VideoThread(QtCore.QThread):
    frame_updated = QtCore.pyqtSignal(QtGui.QImage)
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent  # Reference to main window (for mode and lock flags)
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3, 640)
        self.capture.set(4, 480)
        self.detector = htm.handDetector(detectCon=0.7)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            # Draw header overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (frame.shape[1],40), (0,0,0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            mode = self.parent.mode  # Get current mode

            if mode is None:
                draw_text_with_bg(frame, "Main Menu: Use Global Hotkeys", (10, 30))
            elif mode == 1:
                # Brightness Mode
                frame = self.detector.findHands(frame)
                lmList = self.detector.findPosition(frame, draw=False)
                if lmList:
                    x1, y1 = lmList[4][1], lmList[4][2]
                    x2, y2 = lmList[8][1], lmList[8][2]
                    cv2.circle(frame, (x1, y1), 15, (255,0,255), cv2.FILLED)
                    cv2.circle(frame, (x2, y2), 15, (255,0,255), cv2.FILLED)
                    cv2.line(frame, (x1, y1), (x2, y2), (255,0,255), 3)
                    length = math.hypot(x2-x1, y2-y1)
                    bright = np.interp(length, [50,230], [self.parent.minBright, self.parent.maxBright])
                    brightPer = np.interp(length, [50,300], [0,100])
                    if not self.parent.brightness_locked:
                        sbc.set_brightness(bright)
                        self.parent.brightness_set_msg = f"Brightness: {int(brightPer)}%"
                    else:
                        self.parent.brightness_set_msg = "Brightness locked!"
                    draw_text_with_bg(frame, self.parent.brightness_set_msg, (10,70))
                    draw_slider(frame, (frame.shape[1]-60, 50), (30,300), brightPer, slider_color=(0,255,255))
                draw_text_with_bg(frame, "Brightness Control Mode", (10,30))
            elif mode == 2:
                # Volume Mode
                frame = self.detector.findHands(frame)
                lmList = self.detector.findPosition(frame, draw=False)
                if lmList:
                    x1, y1 = lmList[4][1], lmList[4][2]
                    x2, y2 = lmList[8][1], lmList[8][2]
                    cv2.circle(frame, (x1, y1), 15, (255,0,255), cv2.FILLED)
                    cv2.circle(frame, (x2, y2), 15, (255,0,255), cv2.FILLED)
                    cv2.line(frame, (x1, y1), (x2, y2), (255,0,255), 3)
                    length = math.hypot(x2-x1, y2-y1)
                    vol = np.interp(length, [50,230], [self.parent.minVol, self.parent.maxVol])
                    volPer = np.interp(length, [50,300], [0,100])
                    if not self.parent.volume_locked:
                        if is_mac:
                            set_volume_mac(volPer)
                            self.parent.volume_set_msg = f"Volume: {int(volPer)}%"
                        elif is_windows:
                            self.parent.volume_interface.SetMasterVolumeLevel(vol, None)
                            self.parent.volume_set_msg = f"Volume: {int(vol)}"
                    draw_text_with_bg(frame, self.parent.volume_set_msg, (frame.shape[1]-180,40))
                    draw_slider(frame, (frame.shape[1]-60, 50), (30,300), volPer, slider_color=(0,255,0))
                draw_text_with_bg(frame, "Volume Control Mode", (10,30))
                if self.parent.volume_locked:
                    draw_text_with_bg(frame, "🔒 VOLUME LOCKED", (10,70))
            elif mode == 3:
                # Screenshot Mode
                frame = self.detector.findHands(frame)
                lmList = self.detector.findPosition(frame, draw=False)
                if lmList and self.parent.recognize_gesture(lmList) and not self.parent.screenshot_cooldown:
                    self.parent.take_screenshot()
                if self.parent.screenshot_cooldown:
                    elapsed = time.time() - self.parent.pause_start_time
                    if elapsed < self.parent.pause_duration:
                        draw_text_with_bg(frame, f"Pausing for {self.parent.pause_duration - int(elapsed)} sec", (10,70))
                    else:
                        self.parent.screenshot_cooldown = False
                        draw_text_with_bg(frame, "Ready for new gesture.", (10,70))
                draw_text_with_bg(frame, "Screenshot Mode: Open-hand gesture", (10,30))

            mode_text = f"Mode: {mode}" if mode is not None else "Mode: Main Menu"
            draw_text_with_bg(frame, mode_text, (10, frame.shape[0]-50))
            
            # Convert frame to QImage and emit signal
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QtGui.QImage(rgb_image.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
            self.frame_updated.emit(qt_image)
            time.sleep(0.02)
    
    def stop(self):
        self.running = False
        self.capture.release()

# ------------------------------
# Global Hotkey Listener Thread
# ------------------------------
class HotkeyListenerThread(QtCore.QThread):
    # Signal to update mode (1: brightness, 2: volume, 3: screenshot)
    mode_changed = QtCore.pyqtSignal(int)
    
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.running = True
        self.listener = None

    def run(self):
        # Define global hotkeys using the proper angle bracket syntax for modifiers:
        hotkeys = {
            '<ctrl>+9': lambda: self.mode_changed.emit(1),
            '<ctrl>+8': lambda: self.mode_changed.emit(2),
            '<ctrl>+7': lambda: self.mode_changed.emit(3),
            '<ctrl>+<shift>+v': lambda: self.toggle_volume_lock(),
            '<ctrl>+<shift>+b': lambda: self.toggle_brightness_lock()
        }
        self.listener = keyboard.GlobalHotKeys(hotkeys)
        self.listener.start()
        while self.running:
            time.sleep(0.1)
    
    def toggle_volume_lock(self):
        self.main_app.volume_locked = not self.main_app.volume_locked
        if self.main_app.volume_locked:
            self.main_app.volume_set_msg = "Volume locked!"
        else:
            self.main_app.volume_set_msg = "Volume unlocked, adjust now."
    
    def toggle_brightness_lock(self):
        self.main_app.brightness_locked = not self.main_app.brightness_locked
        if self.main_app.brightness_locked:
            self.main_app.brightness_set_msg = "Brightness locked!"
        else:
            self.main_app.brightness_set_msg = "Brightness unlocked, adjust now."
    
    def stop(self):
        self.running = False
        if self.listener:
            self.listener.stop()

# ------------------------------
# Main Application Window
# ------------------------------
class HandControlApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Control Desktop App")
        self.setGeometry(100, 100, 800, 600)
        
        # Lock flags and messages for each mode
        self.brightness_locked = False
        self.volume_locked = False
        self.screenshot_cooldown = False
        
        self.brightness_set_msg = ""
        self.volume_set_msg = ""
        
        self.pause_duration = 10  # Cooldown in seconds for screenshots
        self.pause_start_time = 0
        
        # Mode: None (main menu), 1 (brightness), 2 (volume), 3 (screenshot)
        self.mode = None
        
        self.minBright = 0
        self.maxBright = 100
        self.minVol, self.maxVol = 0, 100
        
        # Windows volume control initialization
        if is_windows:
            self.volume_interface = initialize_windows_audio()
            volRange = self.volume_interface.GetVolumeRange()
            self.minVol = volRange[0]
            self.maxVol = volRange[1]
        
        # Setup UI
        self.layout = QtWidgets.QVBoxLayout(self)
        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.layout.addWidget(self.video_label)
        
        self.btn_layout = QtWidgets.QHBoxLayout()
        self.btn_brightness = QtWidgets.QPushButton("Brightness Control")
        self.btn_volume = QtWidgets.QPushButton("Volume Control")
        self.btn_screenshot = QtWidgets.QPushButton("Screenshot Mode")
        self.btn_main = QtWidgets.QPushButton("Main Menu")
        self.btn_layout.addWidget(self.btn_brightness)
        self.btn_layout.addWidget(self.btn_volume)
        self.btn_layout.addWidget(self.btn_screenshot)
        self.btn_layout.addWidget(self.btn_main)
        self.layout.addLayout(self.btn_layout)
        
        # Manual mode switching (optional)
        self.btn_brightness.clicked.connect(lambda: self.set_mode(1))
        self.btn_volume.clicked.connect(lambda: self.set_mode(2))
        self.btn_screenshot.clicked.connect(lambda: self.set_mode(3))
        self.btn_main.clicked.connect(lambda: self.set_mode(None))
        
        # Start video processing thread
        self.video_thread = VideoThread(self)
        self.video_thread.frame_updated.connect(self.update_image)
        self.video_thread.start()
        
        # Start global hotkey listener thread
        self.hotkey_listener_thread = HotkeyListenerThread(self)
        self.hotkey_listener_thread.mode_changed.connect(self.set_mode)
        self.hotkey_listener_thread.start()
    
    def set_mode(self, mode):
        self.mode = mode
    
    def update_image(self, qt_image):
        pixmap = QtGui.QPixmap.fromImage(qt_image).scaled(self.video_label.width(),
                                                            self.video_label.height(),
                                                            QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)
    
    def recognize_gesture(self, lmList):
        # Recognize an open-hand gesture (for screenshot mode)
        if len(lmList) == 21:
            thumb_tip = lmList[4][2]
            index_tip = lmList[8][2]
            middle_tip = lmList[12][2]
            ring_tip = lmList[16][2]
            pinky_tip = lmList[20][2]
            thumb_mcp = lmList[2][2]
            index_dip = lmList[7][2]
            middle_dip = lmList[11][2]
            ring_dip = lmList[15][2]
            pinky_dip = lmList[19][2]
            if (thumb_tip < thumb_mcp and index_tip < index_dip and
                middle_tip < middle_dip and ring_tip < ring_dip and
                pinky_tip < pinky_dip):
                return True
        return False
    
    def take_screenshot(self):
        if not self.screenshot_cooldown:
            print("Taking screenshot (triggered by gesture)...")
            self.screenshot_cooldown = True
            self.pause_start_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            screenshot_path = os.path.join(os.path.expanduser("~/Desktop"), f"screenshot_{timestamp}.png")
            screenshot = pyautogui.screenshot()
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            cv2.imwrite(screenshot_path, screenshot)
            print(f"Screenshot saved at {screenshot_path}")
    
    def closeEvent(self, event):
        self.video_thread.stop()
        self.hotkey_listener_thread.stop()
        event.accept()

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = HandControlApp()
    window.show()
    sys.exit(app.exec_())
