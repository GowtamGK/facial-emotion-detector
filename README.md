# 🎭 Real-Time Facial Emotion Detector

A Python application that uses your webcam to detect faces and recognize emotions in real-time!

## Detected Emotions
- 😊 **Happy**
- 😢 **Sad**
- 😠 **Angry**
- 😲 **Surprise**
- 😨 **Fear**
- 🤢 **Disgust**
- 😐 **Neutral**

## Installation

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the emotion detector:
```bash
python emotion_detector.py
```

- Look at your webcam
- Your face will be detected and outlined
- The detected emotion will be shown above your face
- A side panel shows all emotion scores as progress bars
- Press **'Q'** to quit

## Features

- ✅ Real-time face detection using MTCNN
- ✅ 7 emotion categories
- ✅ Visual progress bars for each emotion
- ✅ Color-coded emotions
- ✅ Mirror mode (flipped horizontally)
- ✅ Confidence percentages

## Troubleshooting

- **Webcam not opening?** Make sure no other app is using your camera
- **Slow performance?** The app analyzes every 3rd frame for better speed
- **No face detected?** Ensure good lighting and face the camera directly

## Requirements

- Python 3.8+
- Webcam
- Windows/Mac/Linux
