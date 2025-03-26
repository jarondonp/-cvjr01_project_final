# Advanced Face & Feature Detection App

A comprehensive facial analysis system built with Python, offering multiple capabilities including face detection, recognition, and feature analysis.

## System Requirements

- Python 3.10 or higher
- CMake (required for certain dependencies)
- Windows 10/11, Linux, or macOS
- Webcam (optional, for real-time detection)
- Minimum 4GB RAM (8GB recommended)
- GPU support (optional, improves performance)

## Installation Guide

### Important Note
The installation order of dependencies is crucial for the application to work correctly. Please follow the steps exactly as outlined below.

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/face-detection-app.git
cd face-detection-app
```

2. **Set Up Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

3. **Install Dependencies (Critical Order)**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# 1. Install base dependencies first
pip install protobuf==3.20.0
pip install setuptools>=65.5.0
pip install six>=1.17.0

# 2. Install TensorFlow and core dependencies
pip install tensorflow==2.8.0
pip install numpy==1.26.0

# 3. Install Computer Vision and Machine Learning packages
pip install deepface==0.0.79
pip install opencv-python-headless==4.11.0.86
pip install Pillow==10.4.0
pip install scikit-learn==1.6.1
pip install scipy==1.15.2

# 4. Install UI and visualization packages
pip install streamlit==1.31.0
pip install matplotlib==3.7.1

# 5. Install remaining dependencies
pip install pandas==2.2.3
pip install python-dateutil==2.9.0.post0
pip install python-dotenv==1.0.0
pip install requests==2.32.3
```

## Known Issues and Solutions

### 1. Protobuf Version Conflicts
If you encounter protobuf-related errors:
- Ensure you have installed protobuf==3.20.0 before installing TensorFlow
- Do not upgrade protobuf as it may break TensorFlow compatibility

### 2. Image Display Issues
If you encounter "use_container_width" errors in Streamlit:
- This is a known issue with certain Streamlit versions
- The application has been updated to handle this gracefully

### 3. Face Recognition Model Loading
If face recognition models fail to load:
- Ensure you have a stable internet connection for the first run
- Models will be downloaded and cached automatically
- Check your firewall settings if downloads fail

### 4. Memory Usage
- The application may use significant memory when processing high-resolution images
- Recommended to close other memory-intensive applications
- Consider reducing image resolution for better performance

## Configuration

### Streamlit Configuration (.streamlit/config.toml)
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
enableCORS = false
enableXsrfProtection = false

[runner]
# Uses the Python interpreter from your virtual environment
pythonPath = "venv/Scripts/python.exe"

[browser]
serverAddress = "localhost"
serverPort = 8501

[logger]
level = "info"
```

## Running the Application

1. **Start the Application**
```bash
python -m streamlit run streamlit_app.py
```

2. **Access the Interface**
- Open your web browser
- Navigate to `http://localhost:8501`

## Features

- **Face Detection**: Real-time face detection using OpenCV DNN
- **Feature Recognition**: Detect facial features (eyes, smiles)
- **Face Comparison**: Compare faces between images with similarity analysis
- **Face Recognition**: Register and identify faces using DeepFace
- **Multi-model Analysis**: Uses multiple embedding models for improved accuracy

## Troubleshooting

### Common Issues and Solutions

1. **CMake Not Found**
   - Ensure CMake is properly installed and added to PATH
   - Restart your terminal after installing CMake

2. **DeepFace Installation Issues**
   - Make sure TensorFlow is installed first
   - Verify Python version compatibility

3. **Webcam Access Issues**
   - Check webcam permissions in your system settings
   - Ensure no other application is using the webcam

4. **Version Conflicts**
   - Follow the exact versions in requirements.txt
   - Create a fresh virtual environment if issues persist

## Project Structure

```
face-detection-app/
├── streamlit_app.py      # Main application file
├── requirements.txt      # Project dependencies
├── README.md            # Documentation
└── venv/               # Virtual environment directory
```

## Version Information

Current tested versions of key dependencies:
- Python: 3.10
- numpy: 1.26.0
- tensorflow: 2.8.0
- deepface: 0.0.79
- opencv-python-headless: 4.11.0.86
- streamlit: 1.31.0

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit and Python
- Uses OpenCV for computer vision operations
- Implements DeepFace for facial recognition
- Special thanks to all contributors

## Features

- **Face Detection**: Uses OpenCV DNN with a pre-trained SSD MobileNet model for accurate face detection with adjustable confidence thresholds.
- **Face Recognition**: Register faces and identify them in new images or real-time video. The system stores facial embeddings for multiple models.
- **Facial Analysis**: Detects facial attributes including age, gender, and emotion using DeepFace's pre-trained models.
- **Feature Detection**: Identifies additional facial features including eyes and smiles using Haar Cascade Classifiers with adjustable parameters.
- **Comparison Mode**: Compare faces between two images using two methods:
  - HOG (Histograms of Oriented Gradients) - fast and effective for quick comparisons
  - Embeddings (deep neural networks) - slower but more precise for accurate matching
- **Video Processing**: Process uploaded videos to detect faces and facial features with frame-by-frame analysis.
- **Interactive UI**: Multiple UI components including sliders, tabs, expandable sections, and dynamic metric displays.
- **Performance Metrics**: View processing time and detection statistics for optimization and insight.
- **Downloadable Results**: Export processed images and videos with annotations for further use.
- **Registered Faces Management**: Descriptive table with the ability to delete individual records or all faces at once.

## Implementation Details

### Technologies Used

- **Streamlit**: For the web interface and interactive components, providing a responsive UI
- **OpenCV**: For computer vision operations including face and feature detection using DNN and Haar Cascades
- **NumPy**: For efficient array operations and numerical computations
- **PIL (Pillow)**: For image processing, manipulation, and format conversion
- **DeepFace**: For advanced facial recognition and attribute analysis using deep learning models
- **TensorFlow**: For deep neural network models that power the facial recognition system
- **scikit-learn**: For machine learning algorithms and similarity comparison using cosine distance
- **pandas**: For tabular data handling and management of facial databases

### Model Information

- **Face Detection**: OpenCV DNN module with SSD MobileNet, trained on the WIDER FACE dataset
- **Face Recognition**: Pre-trained DeepFace models:
  - VGG-Face: Based on the VGG-16 architecture, trained on a large-scale facial recognition dataset
  - Facenet: Google's FaceNet model, using a deep convolutional network
  - OpenFace: An open-source facial recognition model based on FaceNet architecture
  - ArcFace: State-of-the-art facial recognition model using Additive Angular Margin Loss
- **Eye Detection**: Haar Cascade Classifier (haarcascade_eye.xml), trained on positive and negative eye images
- **Smile Detection**: Haar Cascade Classifier (haarcascade_smile.xml), for detecting smiles with adjustable parameters

### User Interface Enhancements

The application includes several UI improvements:
- Tab-based navigation for different functionalities, allowing easy switching between modes
- Sidebar with control settings for global parameters that affect all modes
- Progress bars for video processing to show completion status
- Metric displays for performance statistics, including processing time and detection counts
- Expandable sections for detailed information, reducing visual clutter
- Color pickers for customizing display options like bounding box colors
- Descriptive table for managing registered faces with clear information on embeddings and models

## Advanced Features

### Improved Similarity Algorithm
The application includes an enhanced similarity algorithm for facial comparison that:
- Uses a stronger power curve (1.3) for better discrimination between similar and dissimilar faces
- Gives more weight to precise facial structure (25%) for improved matching accuracy
- Applies more aggressive reductions for low similarities to better differentiate non-matches
- Introduces a "critical difference score" that can reduce similarity by up to 25% when significant differences are detected

### Updated Similarity Thresholds
New, stricter similarity ranges for more accurate matching:
- HIGH (80-100%): Very likely the same person
- MEDIUM (65-80%): Possibly the same person
- LOW (35-65%): Unlikely to be the same person
- VERY LOW (0-35%): Different people

### Face Database Management
- Registration of multiple embeddings per person for improved recognition across different conditions
- Support for different embedding models (VGG-Face, Facenet, OpenFace, ArcFace)
- Interactive face management interface with tabular display
- Functionality to delete individual records or all faces at once for database maintenance

## Future Enhancements

- Improve facial recognition accuracy with ensemble methods combining multiple models
- Add face tracking in videos for consistent identification across frames
- Implement user authentication for secure access to the face database
- Add capability to export/import the face database for backup and transfer
- Enhance result visualization with advanced graphs and interactive reports
- Implement batch processing for large datasets of images or videos

## Credits

- Based on the tutorial by Dr. Ernesto Lee
- Uses pre-trained models from OpenCV and DeepFace
- Built with Streamlit and Python

