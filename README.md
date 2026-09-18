# Detection-of-deepfake-videos-on-social-media-platforms
🎥 Deepfake Detection System

A deep learning-based Deepfake Detection System that analyzes videos and classifies them as REAL or FAKE using a Swin Transformer-based architecture. The system extracts visual and audio features from videos and provides a prediction along with a confidence score and detection reason.

📌 Project Overview

Deepfake technology can create highly realistic manipulated videos by altering a person's face, expressions, voice, or lip movements. This project aims to provide an automated system for identifying such manipulated content.

The system uses a Swin Transformer for extracting visual features from video frames and a CNN-based audio network for extracting audio features. These features are combined using a fusion layer to perform the final classification.

The project also provides a localhost web interface and a Telegram bot for submitting videos for analysis.

✨ Features
🎬 Upload a video for analysis
📹 Record a video using the browser camera
🧠 Swin Transformer-based visual analysis
🔊 Audio feature extraction using Mel Spectrograms
🔗 Visual and audio feature fusion
📊 REAL / FAKE classification
📈 Confidence score
📝 Detection reason
🌐 Localhost web application
🤖 Telegram bot integration
🧠 Model Architecture

The detection model consists of three main components:

1. Swin Transformer

The Swin Transformer processes selected video frames and extracts important visual features.

It uses:

Image resizing
Patch partitioning
Window-based self-attention
Shifted window attention
Hierarchical feature extraction
2. Audio CNN

The audio portion processes the extracted audio using a CNN.

The audio is converted into a Mel Spectrogram, which represents the frequency characteristics of the audio signal.

3. Feature Fusion

The visual features from the Swin Transformer and audio features from the CNN are combined.

The fused features are passed through fully connected layers to produce two outputs:

REAL
FAKE
🔄 Processing Workflow
Video Input
     ↓
Video Preprocessing
     ↓
Frame Extraction ──────→ Swin Transformer
     ↓                         ↓
Audio Extraction ───────→ Audio CNN
     ↓                         ↓
     └──────── Feature Fusion ─┘
                  ↓
          Classification Layer
                  ↓
          REAL / FAKE Result
                  ↓
       Confidence + Reason
🛠️ Technologies Used
Technology	Purpose
Python	Main programming language
PyTorch	Deep learning framework
Swin Transformer	Visual feature extraction
Torchvision	Image processing and transformations
Librosa	Audio processing
OpenCV	Video and frame processing
NumPy	Numerical operations
Pillow	Image handling
Flask	Local backend/API
Flask-CORS	Frontend-backend communication
FFmpeg	Video/audio conversion
JavaScript	Frontend interaction
HTML/CSS	Web interface
Telegram Bot API	Telegram-based detection
📊 Input and Output
Input

The system accepts video files through:

Web application upload
Browser camera recording
Telegram bot
Output

The system provides:

Result: FAKE
Confidence: 99.16%
Reason: Audio-visual manipulation detected

For a real video:

Result: REAL
Confidence: 95.40%
Reason: No strong deepfake artifacts detected
🌐 Web Application

The project provides a simple browser-based interface where users can:

Upload a video.
Start a camera recording.
Submit the video for analysis.
View the detection result.
View the confidence score and reason.
Access the Telegram bot.

The application runs locally through localhost.

🤖 Telegram Bot

The project also provides a Telegram interface for convenient detection.

Users can:

Open the Telegram bot.
Send a video.
The bot sends the video to the local detection backend.
The model analyzes the video.
The bot returns the result and confidence.
🎯 Objective

The primary objective of this project is to develop an automated deepfake detection system capable of analyzing both visual and audio information from videos and providing an understandable detection result.

🚀 Future Enhancements
Real-time video detection
Improved explainability using visualization techniques
Larger and more diverse datasets
Support for additional deepfake generation methods
Cloud-based deployment
Improved multilingual audio analysis
Continuous model fine-tuning
