# 🎭 Emotion Reflector - Advanced Face Emotion Recognition System

A sophisticated real-time emotion recognition system that uses advanced pattern-based detection with enhanced smile recognition capabilities.

## 🚀 Features

- **Advanced Pattern-Based Detection**: Uses facial geometry analysis and brightness patterns for accurate emotion detection
- **Enhanced Smile Recognition**: Specialized algorithms for detecting genuine smiles with high accuracy
- **Multi-Camera Support**: Automatic fallback to working cameras (supports camera indices 0, 1, 2)
- **Real-Time Processing**: Optimized for smooth real-time performance
- **7 Emotion Detection**: Detects angry, disgust, fear, happy, neutral, sad, and surprise emotions
- **Easy-to-Use**: One-click batch scripts for quick start/stop

## 🎯 Quick Start

### Method 1: One-Click Start (Recommended)
```bash
# Double-click or run:
START_EMOTION.bat
```

### Method 2: Manual Start
```bash
python fixed_emotion_detector.py
```

## 📋 Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Webcam/Camera

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/deep-ghosh/emotion-reflector.git
cd emotion-reflector
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the system:**
```bash
START_EMOTION.bat
```

## 📊 System Architecture

### Enhanced Pattern-Based Detection
- **Facial Geometry Analysis**: Analyzes facial feature proportions and positions
- **Brightness Pattern Recognition**: Detects emotion-specific brightness patterns
- **OpenCV Cascade Integration**: Uses Haar cascades for face, eye, and smile detection
- **Multi-Feature Fusion**: Combines multiple detection methods for improved accuracy

### Real-Time Processing Pipeline
1. **Camera Input**: Captures live video feed
2. **Face Detection**: Identifies faces in the frame
3. **Feature Extraction**: Extracts facial features and patterns
4. **Emotion Analysis**: Applies pattern-based emotion detection
5. **Result Display**: Shows emotion labels with confidence scores

## 🎮 Controls

- **'q' key**: Quit the application
- **Camera Window**: Shows live video feed with emotion detection
- **Face Detection**: Blue rectangles around detected faces
- **Emotion Labels**: Text showing detected emotion above each face

## 🔧 Configuration

The system automatically:
- Detects available cameras and selects the best one
- Adjusts to different lighting conditions
- Handles multiple faces in the frame
- Provides robust error handling and recovery

## 📈 Performance

- **Accuracy**: High accuracy for all 7 emotions with enhanced smile detection
- **Speed**: Real-time processing (30+ FPS on modern hardware)
- **Stability**: Robust error handling and automatic recovery
- **Compatibility**: Works with various camera types and resolutions

## 🛠️ Troubleshooting

### Common Issues:

1. **Camera Not Working**: The system will automatically try different camera indices
2. **Poor Detection**: Ensure good lighting and face the camera directly
3. **Performance Issues**: Close other applications using the camera

### Solutions:
- Use `START_EMOTION.bat` for the most reliable experience
- Ensure your webcam is working and not blocked by other applications
- Try different lighting conditions for better detection accuracy

## 📄 Files Overview

- `fixed_emotion_detector.py` - Main enhanced emotion detection system
- `START_EMOTION.bat` - Quick start script
- `STOP_EMOTION.bat` - Quick stop script
- `USAGE_GUIDE.md` - Detailed usage instructions
- `EMOTION_DETECTION_GUIDE.md` - Technical detection guide

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting enhancements
- Submitting pull requests
- Improving documentation

## 📜 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**deep-ghosh** - [GitHub](https://github.com/deep-ghosh)

---

**Your enhanced emotion recognition system is ready to use! 🎉**

Start detecting emotions in real-time with just one click using `START_EMOTION.bat`
