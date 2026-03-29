# Tie & Bowtie Detector

A web application that detects **ties** and **bowties** in images using a YOLOv8 object detection model, served with Flask and containerized with Docker.

---

## About the model

The model was trained on a custom dataset of ~200 wedding images collected from Roboflow.
It detects two classes:
-  **Tie**
-  **Bowtie**

It was trained using YOLOv8s for 20 epochs and achieved:
- **96.1% mAP50** on the test set
- **99.1%** accuracy for bowtie detection
- **93.1%** accuracy for tie detection

---

## How to Run Locally with Docker

**1. Clone the repository:**
```bash
git clone https://github.com/marcsawayai/tie-bowtie-app.git
cd tie-bowtie-app
```

**2. Build the Docker image:**
```bash
docker build -t tie-bowtie-app .
```

**3. Run the container:**
```bash
docker run -p 5000:5000 tie-bowtie-app
```

**4. Open your browser and go to:**
```
http://127.0.0.1:5000
```

---

##  How to Use the Interface

1. Open the app in your browser
2. Paste a direct image URL in the input box
3. Click **Detect**
4. The app will display the image and show the detected class with confidence score

---

##  Dependencies

- Python 3.10
- Flask
- Ultralytics YOLOv8
- OpenCV
- Requests

---

## 👤 Author

**Marc Sawaya**  
AIC - Machine Learning Specialization  
