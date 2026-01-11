"""
Flask Web Server for Emotion Detection
Uses the FER library for accurate emotion recognition
"""

from flask import Flask, render_template, Response, jsonify
import cv2
from fer import FER
import threading
import time

app = Flask(__name__)

# Global variables
detector = None
camera = None
current_emotions = {}
current_face_box = None
dominant_emotion = None
confidence = 0
is_running = False
lock = threading.Lock()

def initialize_detector():
    """Initialize the FER detector"""
    global detector
    print("Loading emotion detection model...")
    detector = FER(mtcnn=True)
    print("Model loaded successfully!")

def get_camera():
    """Get or create camera instance"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def release_camera():
    """Release camera"""
    global camera
    if camera is not None:
        camera.release()
        camera = None

# Emotion colors (BGR for OpenCV)
EMOTION_COLORS = {
    'happy': (0, 255, 255),      # Yellow
    'sad': (255, 165, 0),         # Orange-ish Blue
    'angry': (0, 0, 255),         # Red
    'surprise': (255, 0, 255),    # Magenta
    'fear': (128, 0, 128),        # Purple
    'disgust': (0, 200, 0),       # Green
    'neutral': (200, 200, 200)    # Gray
}

def generate_frames():
    """Generate video frames with emotion detection"""
    global current_emotions, current_face_box, dominant_emotion, confidence, is_running
    
    cam = get_camera()
    if not cam.isOpened():
        return
    
    is_running = True
    frame_count = 0
    
    while is_running:
        success, frame = cam.read()
        if not success:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect emotions every 3 frames for performance
        if frame_count % 3 == 0:
            try:
                result = detector.detect_emotions(frame)
                
                with lock:
                    if result:
                        face = result[0]
                        current_emotions = face['emotions']
                        current_face_box = face['box']
                        dominant_emotion = max(current_emotions, key=current_emotions.get)
                        confidence = current_emotions[dominant_emotion]
                    else:
                        current_emotions = {}
                        current_face_box = None
                        dominant_emotion = None
            except Exception as e:
                print(f"Detection error: {e}")
        
        # Draw on frame
        with lock:
            if current_face_box:
                x, y, w, h = current_face_box
                color = EMOTION_COLORS.get(dominant_emotion, (255, 255, 255))
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Draw emotion label
                if dominant_emotion:
                    label = f"{dominant_emotion.upper()} {int(confidence * 100)}%"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 10, y - 5), color, -1)
                    cv2.putText(frame, label, (x + 5, y - 12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        frame_count += 1
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotions')
def get_emotions():
    """API endpoint to get current emotions"""
    with lock:
        return jsonify({
            'emotions': current_emotions,
            'dominant': dominant_emotion,
            'confidence': confidence,
            'face_detected': current_face_box is not None
        })

@app.route('/stop')
def stop():
    """Stop the camera"""
    global is_running
    is_running = False
    release_camera()
    return jsonify({'status': 'stopped'})

@app.route('/start')
def start():
    """Start/restart the camera"""
    global is_running
    is_running = True
    return jsonify({'status': 'started'})

if __name__ == '__main__':
    initialize_detector()
    print("\n" + "="*50)
    print("  EMOTION DETECTOR WEB SERVER")
    print("  Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
