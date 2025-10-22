import cv2
import mediapipe as mp
import numpy as np
import socket
import time
import threading
from collections import deque

class GestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # ESP32 communication setup
        self.esp32_ip = "192.168.1.100"  # Replace with actual ESP32 IP
        self.esp32_port = 80
        self.socket = None
        
        # Gesture recognition variables
        self.current_gesture = "stop"
        self.gesture_buffer = deque(maxlen=5)  # Buffer for gesture smoothing
        self.last_command_time = time.time()
        self.command_cooldown = 0.5  # Minimum time between commands
        
        # Initialize socket connection
        self.connect_to_esp32()
    
    def connect_to_esp32(self):
        """Establish connection to ESP32"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.esp32_ip, self.esp32_port))
            print(f"Connected to ESP32 at {self.esp32_ip}:{self.esp32_port}")
        except Exception as e:
            print(f"Failed to connect to ESP32: {e}")
            self.socket = None
    
    def send_command(self, command):
        """Send command to ESP32"""
        if self.socket and time.time() - self.last_command_time > self.command_cooldown:
            try:
                self.socket.send(command.encode())
                self.last_command_time = time.time()
                print(f"Sent command: {command}")
            except Exception as e:
                print(f"Failed to send command: {e}")
                self.connect_to_esp32()  # Try to reconnect
    
    def calculate_finger_angles(self, landmarks):
        """Calculate angles between fingers to determine gesture"""
        # Get landmark coordinates
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # Calculate if fingers are extended
        fingers_up = []
        
        # Thumb (special case - check x coordinate)
        if thumb_tip.x > thumb_ip.x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
        
        # Other fingers (check y coordinate)
        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        finger_pips = [index_pip, middle_pip, ring_pip, pinky_pip]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if tip.y < pip.y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        return fingers_up
    
    def recognize_gesture(self, fingers_up):
        """Recognize gesture based on finger positions"""
        # Define gesture patterns
        gestures = {
            "stop": [0, 0, 0, 0, 0],      # Closed fist
            "forward": [0, 1, 0, 0, 0],    # Index finger up
            "backward": [0, 1, 1, 0, 0],   # Index and middle finger up
            "left": [1, 0, 0, 0, 0],       # Thumb up
            "right": [0, 0, 0, 0, 1],      # Pinky up
            "fast": [1, 1, 1, 1, 1]        # All fingers up
        }
        
        # Find matching gesture
        for gesture_name, pattern in gestures.items():
            if fingers_up == pattern:
                return gesture_name
        
        return "unknown"
    
    def smooth_gesture(self, gesture):
        """Apply smoothing to reduce gesture jitter"""
        self.gesture_buffer.append(gesture)
        
        # Return most common gesture in buffer
        if len(self.gesture_buffer) >= 3:
            gesture_counts = {}
            for g in self.gesture_buffer:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
            
            return max(gesture_counts, key=gesture_counts.get)
        
        return gesture
    
    def process_frame(self, frame):
        """Process a single frame for gesture recognition"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        gesture = "stop"  # Default gesture
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Calculate finger positions
                fingers_up = self.calculate_finger_angles(hand_landmarks.landmark)
                
                # Recognize gesture
                gesture = self.recognize_gesture(fingers_up)
                
                # Apply smoothing
                gesture = self.smooth_gesture(gesture)
                
                # Display gesture on frame
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Send command if gesture changed
                if gesture != self.current_gesture:
                    self.current_gesture = gesture
                    self.send_command(gesture)
        
        return frame
    
    def run(self):
        """Main loop for gesture recognition"""
        print("Starting gesture recognition...")
        print("Gestures:")
        print("- Closed fist: STOP")
        print("- Index finger: FORWARD")
        print("- Index + Middle: BACKWARD")
        print("- Thumb: LEFT")
        print("- Pinky: RIGHT")
        print("- All fingers: FAST MODE")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Display frame
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        if self.socket:
            self.socket.close()

if __name__ == "__main__":
    recognizer = GestureRecognizer()
    recognizer.run()

