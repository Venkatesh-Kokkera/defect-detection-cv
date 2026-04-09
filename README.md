👁️ Real-Time Defect Detection System

Two-stage computer vision pipeline using YOLOv8 + EfficientNet-B4 for manufacturing quality inspection — achieving +20% defect detection accuracy over the previous baseline.

Show Image
Show Image
Show Image
Show Image
Show Image
Show Image

🎯 Problem
Manual visual inspection on manufacturing lines is slow, inconsistent, and misses micro-level defects at production speed. This system replaces manual QA with a real-time two-stage CV pipeline that runs at 30+ FPS and classifies 6 defect types with high precision.

✨ Features

Real-Time — YOLOv8 runs at 30+ FPS on standard GPU hardware
Two-Stage Pipeline — YOLO for localization + EfficientNet-B4 for fine-grained classification
Transfer Learning — ImageNet pretrained → fine-tuned on domain defect dataset
6 Defect Classes — scratch, crack, dent, contamination, corrosion, pass
Explainability — SHAP + Grad-CAM visualizations for model decisions
REST API — FastAPI /detect endpoint for single images or batch
MLflow Tracking — All experiments, weights, metrics versioned
Production Deployment — Dockerized, deployed on AWS SageMaker


📊 Results
MetricBaselineThis SystemDefect Detection Accuracy72.3%92.1% (+20%)False Positive Rate18.4%6.2%Inference Speed8 FPS30+ FPSDefect Classes36

🛠️ Tech Stack
ComponentTechnologyDetectionYOLOv8 (Ultralytics)ClassificationEfficientNet-B4 · ResNet-50FrameworkPyTorch 2.xCV LibraryOpenCV 4.xExplainabilitySHAP · Grad-CAMExperiment TrackingMLflowAPIFastAPIDeploymentDocker · AWS SageMaker

🚀 Quick Start
bashgit clone https://github.com/Venkatesh-Kokkera/defect-detection-cv.git
cd defect-detection-cv
pip install -r requirements.txt
python detect.py --source ./data/sample_images/ --conf 0.5
uvicorn app.main:app --host 0.0.0.0 --port 8000

Venkatesh Kokkera · 📧 vkokkeravk@gmail.com · 💼 LinkedIn:https://www.linkedin.com/in/venkatesh-ko/ · 📞 +1 (203) 479-2974 . 📍 Lowell, MA 
