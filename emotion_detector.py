"""
Real-time Facial Emotion Detection using OpenCV and FER
Detects: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
"""

import cv2
from fer import FER
import numpy as np

# Emotion colors (BGR format for OpenCV)
EMOTION_COLORS = {
    'happy': (0, 255, 255),      # Yellow
    'sad': (255, 100, 100),       # Light Blue
    'angry': (0, 0, 255),         # Red
    'surprise': (255, 0, 255),    # Magenta
    'fear': (128, 0, 128),        # Purple
    'disgust': (0, 128, 0),       # Green
    'neutral': (200, 200, 200)    # Gray
}

# Emotion emojis for display
EMOTION_EMOJIS = {
    'happy': ':)',
    'sad': ':(',
    'angry': '>:(',
    'surprise': ':O',
    'fear': 'D:',
    'disgust': ':P',
    'neutral': ':|'
}


def draw_emotion_panel(frame, emotions, x_offset):
    """Draw a side panel showing all emotion scores"""
    panel_width = 250
    panel_height = 300
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_offset, 20), (x_offset + panel_width, panel_height + 20), 
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "EMOTION DETECTOR", (x_offset + 20, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(frame, (x_offset + 10, 70), (x_offset + panel_width - 10, 70), 
             (100, 100, 100), 2)
    
    if emotions:
        y_pos = 100
        for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            emoji = EMOTION_EMOJIS.get(emotion, '')
            
            # Emotion name and emoji
            text = f"{emotion.upper()} {emoji}"
            cv2.putText(frame, text, (x_offset + 15, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Progress bar background
            bar_x = x_offset + 15
            bar_width = 180
            bar_height = 12
            cv2.rectangle(frame, (bar_x, y_pos + 5), 
                         (bar_x + bar_width, y_pos + 5 + bar_height), 
                         (50, 50, 50), -1)
            
            # Progress bar fill
            fill_width = int(bar_width * score)
            cv2.rectangle(frame, (bar_x, y_pos + 5), 
                         (bar_x + fill_width, y_pos + 5 + bar_height), 
                         color, -1)
            
            # Percentage
            cv2.putText(frame, f"{int(score * 100)}%", (bar_x + bar_width + 5, y_pos + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            y_pos += 35
    else:
        cv2.putText(frame, "No face detected", (x_offset + 30, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)


def draw_dominant_emotion(frame, dominant_emotion, confidence, face_box):
    """Draw the dominant emotion above the face"""
    if dominant_emotion:
        x, y, w, h = face_box
        color = EMOTION_COLORS.get(dominant_emotion, (255, 255, 255))
        emoji = EMOTION_EMOJIS.get(dominant_emotion, '')
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Draw emotion label above face
        label = f"{dominant_emotion.upper()} {emoji} ({int(confidence * 100)}%)"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Background for label
        cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 10, y - 5), color, -1)
        cv2.putText(frame, label, (x + 5, y - 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


def main():
    print("=" * 50)
    print("  FACIAL EMOTION DETECTION")
    print("  Press 'Q' to quit")
    print("=" * 50)
    print("\nInitializing emotion detector...")
    
    # Initialize the FER detector (uses MTCNN for face detection)
    detector = FER(mtcnn=True)
    
    print("Opening webcam...")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        print("Make sure your webcam is connected and not in use by another app.")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Webcam opened successfully!")
    print("\nDetecting emotions... Look at the camera!")
    
    frame_count = 0
    emotions = {}
    dominant_emotion = None
    confidence = 0
    face_box = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame!")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Only analyze every 3rd frame for performance
        if frame_count % 3 == 0:
            # Detect emotions
            result = detector.detect_emotions(frame)
            
            if result:
                # Get the first face detected
                face = result[0]
                emotions = face['emotions']
                face_box = face['box']
                
                # Find dominant emotion
                dominant_emotion = max(emotions, key=emotions.get)
                confidence = emotions[dominant_emotion]
            else:
                emotions = {}
                dominant_emotion = None
                face_box = None
        
        # Draw the emotion panel on the right side
        panel_x = frame.shape[1] - 270
        draw_emotion_panel(frame, emotions, panel_x)
        
        # Draw dominant emotion on face
        if face_box:
            draw_dominant_emotion(frame, dominant_emotion, confidence, face_box)
        
        # Draw instructions at bottom
        cv2.putText(frame, "Press 'Q' to quit", (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('Emotion Detector', frame)
        
        frame_count += 1
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nQuitting...")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Emotion detector closed.")


if __name__ == "__main__":
    main()
