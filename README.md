# 🧠 Real-Time Head Pose Detection

A web-based real-time head pose tracking application built with FastAPI, OpenCV, and MediaPipe.

## ✨ Features

- **Real-time head pose tracking** using MediaPipe Face Mesh
- **Web-based interface** with live video streaming
- **Customizable visualization options:**
  - Face detection bounding box
  - Face mesh overlay
  - Nose direction indicator
  - Head pose text display
- **Audio alerts** for head movement detection
- **Responsive UI** with dark/light mode toggle
- **Cross-platform compatibility**

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam
- Modern web browser

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AnantKhandaka/head-pose-tracker.git
cd REPO_NAME
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install fastapi uvicorn opencv-python mediapipe numpy
```

### Running the Application

```bash
uvicorn app:app --reload
```

Open your browser and navigate to `http://127.0.0.1:8000`

## 🎯 Usage

1. Click "Start Video" to begin webcam capture
2. Allow camera permissions when prompted
3. Toggle various visualization options:
   - **Face Box**: Shows face detection boundary
   - **Face Mesh**: Displays facial landmark mesh
   - **Nose Direction**: Shows head orientation line
   - **Direction Text**: Displays head pose information
   - **Alert Sound**: Enables audio notifications

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python
- **Computer Vision**: OpenCV, MediaPipe
- **Frontend**: HTML5, CSS3, JavaScript
- **Streaming**: MJPEG over HTTP

## 📁 Project Structure

```
├── app.py              # Main FastAPI application
├── templates/
│   └── index.html      # Web interface
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for face detection and tracking
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [OpenCV](https://opencv.org/) for computer vision utilities