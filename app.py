"""
Flask Web Server for Emotion Detection
Cloud-ready version - webcam capture happens in browser
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)

# Lazy loading - detector will be initialized on first request
detector = None

def get_detector():
    """Lazy load the FER detector"""
    global detector
    if detector is None:
        print("Loading emotion detection model...")
        from fer import FER
        detector = FER(mtcnn=True)
        print("Model loaded successfully!")
    return detector

# Emotion colors for drawing
EMOTION_COLORS = {
    'happy': (0, 255, 255),
    'sad': (255, 165, 0),
    'angry': (0, 0, 255),
    'surprise': (255, 0, 255),
    'fear': (128, 0, 128),
    'disgust': (0, 200, 0),
    'neutral': (200, 200, 200)
}

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze an image for emotions"""
    try:
        # Get base64 image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Get detector (lazy load)
        det = get_detector()
        
        # Detect emotions
        result = det.detect_emotions(frame)
        
        if result:
            face = result[0]
            emotions = face['emotions']
            box = face['box']
            dominant = max(emotions, key=emotions.get)
            confidence = emotions[dominant]
            
            # Draw on frame
            x, y, w, h = box
            color = EMOTION_COLORS.get(dominant, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Draw label
            label = f"{dominant.upper()} {int(confidence * 100)}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 10, y - 5), color, -1)
            cv2.putText(frame, label, (x + 5, y - 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Encode processed frame back to base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            processed_image = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'emotions': emotions,
                'dominant': dominant,
                'confidence': confidence,
                'face_detected': True,
                'processed_image': f'data:image/jpeg;base64,{processed_image}'
            })
        else:
            # No face detected - return original image
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            processed_image = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'emotions': {},
                'dominant': None,
                'confidence': 0,
                'face_detected': False,
                'processed_image': f'data:image/jpeg;base64,{processed_image}'
            })
            
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy'})

@app.route('/warmup')
def warmup():
    """Endpoint to pre-load the model"""
    get_detector()
    return jsonify({'status': 'model loaded'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*50}")
    print("  EMOTION DETECTOR WEB SERVER")
    print(f"  Open http://localhost:{port} in your browser")
    print('='*50 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
