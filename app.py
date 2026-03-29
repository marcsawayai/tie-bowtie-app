from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import tempfile
import requests
import os

app = Flask(__name__)

# Load model once at startup
model = YOLO("best.pt")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_url = data.get("url", "").strip()

    if not image_url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Download image from URL
        response = requests.get(
            image_url,
            headers={"User-Agent": "Mozilla/5.0"},
            verify=False
        )

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Run YOLO prediction
        results = model(tmp_path)
        detections = results[0].boxes
        os.remove(tmp_path)

        # Parse detections
        output = []
        for box in detections:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = "bowtie" if class_id == 0 else "tie"
            output.append({
                "class": class_name,
                "confidence": round(confidence * 100, 2)
            })

        if not output:
            return jsonify({"message": "No tie or bowtie detected.", "detections": []})

        return jsonify({"message": f"{len(output)} detection(s) found.", "detections": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)