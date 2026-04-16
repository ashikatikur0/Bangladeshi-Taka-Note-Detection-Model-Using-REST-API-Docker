# bangladeshi_note-coin_detect

collab link https://colab.research.google.com/drive/1eNzYwcDC18B6OS1-cimA5P0S06CRQM5k#scrollTo=DvdfAR2zc0tc

📄 Full Project Summary (README / Description)

This project presents a complete deployment pipeline for a Bangladeshi Taka Note Detection system using a YOLO-based deep learning model. The trained model is integrated into a RESTful API built with FastAPI and containerized using Docker for scalable and portable deployment.

The system accepts an input image (JPEG/PNG), performs real-time object detection, and returns structured JSON responses including detected denominations, confidence scores, and bounding box coordinates.

🔍 Key Features
YOLO-based object detection for Bangladeshi currency notes
REST API endpoint (/predict) for image-based inference
JSON response with class labels, confidence, and bounding boxes
Robust error handling for invalid inputs
Dockerized application for easy deployment and reproducibility
Interactive API testing via Swagger UI (/docs)


⚙️ Tech Stack
Python
YOLO (Ultralytics)
FastAPI
OpenCV
Docker



🚀 Workflow
Train YOLO model on Bangladeshi Taka dataset (Phase-1)
Load trained weights and perform inference
Build REST API using FastAPI
Test API using Postman/curl
Containerize application using Docker
(Optional) Deploy to cloud platforms


📦 API Usage
Endpoint: /predict
Method: POST
Input: Image file
Output: JSON with detections

🌐 Deployment

The project can be deployed locally using Docker or on cloud platforms like AWS, Render, or Railway for public API access.
