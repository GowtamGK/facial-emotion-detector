# 🎭 Emotion Detector Web App

A real-time facial emotion detection web application powered by Python, TensorFlow, and the FER (Facial Expression Recognition) library. Features a beautiful modern UI with live webcam feed and emotion analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?logo=opencv)

## ✨ Features

- 🎥 **Real-time webcam feed** with face detection
- 🧠 **Deep learning emotion recognition** using TensorFlow
- 📊 **Live emotion bars** showing confidence levels for all emotions
- 🎨 **Beautiful dark-themed UI** with animated gradients
- 🔄 **Dynamic updates** - emotions change in real-time
- 📱 **Responsive design** - works on desktop and tablets

## 🎭 Detected Emotions

| Emotion | Emoji | Color |
|---------|-------|-------|
| Happy | 😊 | Yellow |
| Sad | 😢 | Blue |
| Angry | 😠 | Red |
| Surprised | 😲 | Pink |
| Fearful | 😨 | Purple |
| Disgusted | 🤢 | Green |
| Neutral | 😐 | Gray |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows/Mac/Linux

### Installation

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/emotion-detector-web.git
   cd emotion-detector-web
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   > ⏱️ First installation may take a few minutes as it downloads TensorFlow and other ML libraries.

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
emotion_detector_web/
├── app.py              # Flask server with emotion detection API
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── templates/
    └── index.html     # Web interface
```

## 🖥️ Usage

1. Open `http://localhost:5000` in your browser
2. Click the **"Start Camera"** button
3. Allow camera permissions when prompted
4. Look at the camera and make different expressions!
5. Watch the emotion bars update in real-time
6. Click **"Stop"** when finished

## 🔧 Configuration

### Change Port
Edit `app.py` and modify the last line:
```python
app.run(host='0.0.0.0', port=5000)  # Change 5000 to your desired port
```

### Detection Frequency
To adjust how often emotions are detected (affects performance), edit `app.py`:
```python
# In generate_frames() function
if frame_count % 3 == 0:  # Analyze every 3rd frame (change this number)
```

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **ML/AI**: TensorFlow, FER (Facial Expression Recognition), MTCNN
- **Computer Vision**: OpenCV
- **Frontend**: HTML5, CSS3, JavaScript
- **Video Streaming**: MJPEG over HTTP

## ⚡ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/video_feed` | GET | MJPEG video stream |
| `/emotions` | GET | JSON with current emotion data |
| `/start` | GET | Start camera |
| `/stop` | GET | Stop camera |

### Example `/emotions` Response
```json
{
  "emotions": {
    "happy": 0.85,
    "sad": 0.02,
    "angry": 0.01,
    "surprise": 0.05,
    "fear": 0.01,
    "disgust": 0.01,
    "neutral": 0.05
  },
  "dominant": "happy",
  "confidence": 0.85,
  "face_detected": true
}
```

## 🐛 Troubleshooting

### Camera not working?
- Make sure no other application is using your webcam
- Try closing other browser tabs that might be using the camera
- Restart the Flask server

### Slow performance?
- Close other applications to free up CPU
- Increase the frame skip value in `app.py` (change `% 3` to `% 5` or higher)
- Reduce browser window size

### ModuleNotFoundError?
- Make sure your virtual environment is activated
- Run `pip install -r requirements.txt` again

### Port already in use?
- Change the port in `app.py` or kill the process using port 5000:
  ```bash
  # Windows
  netstat -ano | findstr :5000
  taskkill /PID <PID> /F
  ```

## 📄 License

MIT License - feel free to use this project for learning and personal projects!

## 🙏 Acknowledgments

- [FER Library](https://github.com/justinshenk/fer) - Facial Expression Recognition
- [TensorFlow](https://tensorflow.org/) - Machine Learning Framework
- [OpenCV](https://opencv.org/) - Computer Vision Library
- [Flask](https://flask.palletsprojects.com/) - Web Framework

---

Made with ❤️ and 🐍 Python
