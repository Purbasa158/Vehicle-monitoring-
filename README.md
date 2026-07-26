# Real-Time Vehicle Detection and Automatic Number Plate Recognition (ANPR)

## Overview

This project implements a real-time Automatic Number Plate Recognition (ANPR) system using a combination of deep learning and classical image processing techniques.

The system detects vehicles from video frames, identifies license plates using a custom-trained YOLOv8 model, enhances the detected plate using OpenCV preprocessing, and extracts the license number using EasyOCR.

The goal of this project is to demonstrate how deep learning and traditional computer vision techniques can be combined to build an efficient and practical vehicle monitoring solution.

---

## Features

- Real-time vehicle detection
- License plate localization
- Automatic license plate recognition (OCR)
- Classical image preprocessing for improved OCR accuracy
- Video-based inference
- Modular pipeline for easy extension

---

## Tech Stack

- Python
- YOLOv8
- OpenCV
- EasyOCR
- NumPy
- Ultralytics

---

## Project Workflow

```text
Input Video
      │
      ▼
Vehicle Detection (YOLOv8)
      │
      ▼
License Plate Detection
      │
      ▼
Image Preprocessing
  • Grayscale
  • Bilateral Filter
  • Adaptive Threshold
      │
      ▼
EasyOCR
      │
      ▼
Recognized License Plate
```

---

## Folder Structure

```text
Vehicle-monitoring/
│
├── anpr/
├── models/
├── assets/
├── input/
├── output/
├── tests/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Image Processing Pipeline

Before performing OCR, the detected license plate is enhanced using OpenCV.

The preprocessing pipeline includes:

- Grayscale conversion
- Bilateral filtering
- Adaptive thresholding

These steps help reduce noise while preserving character edges, leading to better OCR performance.

---

## Applications

- Smart parking systems
- Toll automation
- Traffic monitoring
- Vehicle access control
- Smart city applications

---

## Future Improvements

Some features planned for future versions include:

- Vehicle tracking
- Speed estimation
- Multi-camera support
- Database integration
- REST API
- Web dashboard
- Vehicle counting and analytics

---

## Installation

Clone the repository

```bash
git clone https://github.com/Purbasa158/Vehicle-monitoring.git
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the project

```bash
python runner.py
```

---

## Skills Demonstrated

- Computer Vision
- Deep Learning
- Object Detection
- Optical Character Recognition (OCR)
- Image Processing
- Python
- OpenCV
- YOLOv8
- EasyOCR

---

## Author

**Purbasha Pani**

B.Tech Electronics & Instrumentation Engineering

Odisha University of Technology and Research (OUTR)

---

## Acknowledgements

This project makes use of the following open-source libraries:

- Ultralytics YOLOv8
- OpenCV
- EasyOCR
- NumPy
