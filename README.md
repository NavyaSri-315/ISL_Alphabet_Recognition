# ISL Alphabet Recognition

**An AI-powered Indian Sign Language (ISL) alphabet recognition system using Computer Vision and Machine Learning.**

This system recognizes Indian Sign Language manual alphabet gestures (A–Z) through a webcam and converts them into readable English letters in real time. It combines MediaPipe hand-landmark detection with machine learning classification to provide an interactive sign recognition and learning experience.

## Overview

Traditional image-based recognition systems can be computationally expensive and sensitive to background variations. SignSpeak addresses this challenge by extracting hand landmarks from images and using these coordinates as features for classification rather than training directly on raw image pixels.

The project demonstrates a complete machine learning workflow, from feature extraction and preprocessing to model comparison, real-time inference, and web-based deployment.

## Features

- **Real-Time Recognition:** Recognizes ISL alphabet gestures using a webcam.
- **Hand Landmark Detection:** Uses MediaPipe to detect 21 landmarks per hand.
- **Feature Normalization:** Applies wrist-relative translation and scale normalization.
- **Machine Learning Model Comparison:** Evaluates five classification algorithms to identify a suitable model.
- **Prediction Confidence:** Displays confidence scores when supported by the trained model.
- **Visual Hand Tracking:** Uses OpenCV to display the camera feed and detected hand landmarks.
- **Interactive Learning Interface:** Includes an ISL alphabet reference section with sign-learning guidance and a visual alphabet chart.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core development and ML pipeline |
| OpenCV | Webcam access and image processing |
| MediaPipe | Hand landmark detection |
| NumPy | Numerical operations and coordinate transformations |
| Pandas | Dataset processing and manipulation |
| Scikit-learn | Machine learning model training and evaluation |
| Streamlit | Interactive web interface |
| Pickle | Saving and loading the trained model |

## Machine Learning Workflow

The system follows a structured pipeline:

1. **Feature Extraction:** Processes alphabet images using MediaPipe Hands and extracts hand landmark coordinates.
2. **Data Preparation:** Stores the extracted features and corresponding alphabet labels in a CSV dataset.
3. **Preprocessing:** Normalizes coordinates relative to the wrist and scales them to reduce sensitivity to hand position and size.
4. **Model Training:** Trains and compares five machine learning classifiers.
5. **Model Evaluation:** Evaluates model performance using a training and testing split.
6. **Model Saving:** Saves the selected trained classifier as `isl_model.pkl`.
7. **Real-Time Prediction:** Processes webcam frames, extracts and normalizes landmarks using the same feature pipeline, and predicts the corresponding alphabet.

### Coordinate Normalization

**Translation normalization:** Subtracts the wrist landmark coordinates from the other landmarks to make the representation less dependent on the hand's position in the frame.

**Scale normalization:** Scales the coordinates using the maximum distance from the wrist, helping reduce the influence of hand size and camera distance.

These transformations help the classifier focus on the relative geometry of the hand.

## Machine Learning Algorithms

Five classifiers were trained and compared:

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes

The best-performing model from the evaluation is saved for inference. Model selection should be based on the final evaluation results rather than accuracy alone.

## Application Interface

SignSpeak includes two main sections:

### 1. Live Recognition

- Activates the webcam for live sign recognition.
- Detects hand landmarks and displays the hand skeleton.
- Predicts the corresponding alphabet letter.
- Displays the prediction and confidence score when available.

### 2. Learn ISL Alphabet

- Provides an A–Z reference for manual alphabet signs.
- Offers visual learning guidance for users practising ISL alphabet gestures.

## Project Structure

```text
SignSpeak/
├── app.py
├── extract_features.py
├── preprocess.py
├── train_models.py
├── requirements.txt
├── isl_landmarks.csv
├── isl_landmarks_processed.csv
├── isl_model.pkl
└── README.md
```

*Note: The structure above represents the main files used in the workflow. Dataset folders, generated files, and additional resources may vary in your local repository.*

## Installation and Setup

### Prerequisites

- Python 3.10 or another Python version compatible with the installed MediaPipe and project dependencies
- A working webcam
- Git

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SignSpeak
```

Replace the placeholder URL with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal by Streamlit. Allow camera access when prompted to use live recognition.

## Running the Machine Learning Pipeline

If you want to reproduce the model-building workflow, execute the scripts in the appropriate order:

```bash
python extract_features.py
python preprocess.py
python train_models.py
streamlit run app.py
```

Run these scripts only when the required dataset folders and input files are available. Feature extraction and model training may take time depending on the dataset size and hardware.

The application requires the saved model and the expected feature format to be available before live predictions can work.

## Applications

- Educational demonstrations of Indian Sign Language alphabet recognition
- Interactive sign alphabet learning
- Computer vision and machine learning experimentation
- Prototyping assistive communication technologies

## Limitations

- The current scope is manual alphabet recognition (A–Z), not continuous sign language sentence translation.
- Recognition depends on successful hand detection, image quality, lighting, and hand visibility.
- Performance may vary between the training dataset and real-world webcam conditions.
- The system's output is a machine learning prediction and should not be treated as a guaranteed interpretation of a sign.

## Future Improvements

- Expand recognition to additional ISL gestures and vocabulary.
- Improve robustness across lighting conditions, camera angles, and diverse environments.
- Explore continuous gesture recognition and word-level prediction.
- Evaluate generalization using independent real-world test data.

## Learning Outcomes

Through this project, I explored:

- Computer vision and hand landmark detection
- Feature engineering and coordinate normalization
- Supervised machine learning and classifier comparison
- Model evaluation and serialization
- Real-time webcam inference
- Building an interactive application with Streamlit
- Version control and project documentation using Git and GitHub

## Acknowledgements

- [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) for hand landmark detection
- [OpenCV](https://opencv.org/) for computer vision tools
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [Streamlit](https://streamlit.io/) for the interactive web application framework

---

**Project:** SignSpeak — ISL Alphabet Recognition  
**Domain:** Data Mining and Data Warehousing (DMDW), Machine Learning, Computer Vision  
**Language:** Python
