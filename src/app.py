from flask import Flask, Response, render_template, jsonify
import cv2
import torch
import cnn_models
import joblib
import numpy as np
import os
import sys

# ===== RESOURCE PATH FUNCTION =====
def resource_path(relative_path):
    """
    Get absolute path to resource
    Works for both VS Code and PyInstaller EXE
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        # Get path to root directory, which is one level up from src (where this file is)
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return os.path.join(base_path, relative_path)

# ===== FLASK APP =====
app = Flask(
    __name__,
    template_folder=resource_path("templates")
)

# ===== MODEL + LABEL PATHS =====
model_path = resource_path("outputs/model.pth")
lb_path = resource_path("outputs/lb.pkl")

# ===== LOAD MODEL =====
model = cnn_models.CustomCNN()
model.load_state_dict(
    torch.load(model_path, map_location='cpu')
)
model.eval()

# ===== LOAD LABEL BINARIZER =====
lb = joblib.load(lb_path)

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

# ===== STORE CURRENT PREDICTION =====
current_prediction = ""

# ===== VIDEO FRAME GENERATOR =====
def generate_frames():
    global current_prediction

    while True:
        success, frame = cap.read()

        if not success:
            break

        # mirror effect
        display_frame = cv2.flip(frame, 1)

        # ROI coordinates
        x1, y1 = 100, 100
        x2, y2 = 324, 324

        # extract hand region
        hand = frame[y1:y2, x1:x2]

        # prevent crashes if ROI invalid
        if hand.size == 0:
            continue

        hand = cv2.resize(hand, (224, 224))

        # preview window
        hand_preview = cv2.resize(hand, (100, 100))
        display_frame[10:110, 500:600] = hand_preview

        # preprocess
        image = np.transpose(hand, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image).unsqueeze(0)

        # prediction
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            label = lb.classes_[preds.item()]

        # save latest prediction
        current_prediction = label

        # show prediction
        cv2.putText(
            display_frame,
            f"Prediction: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # encode frame
        _, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

# ===== HOME PAGE =====
@app.route('/')
def index():
    return render_template('index.html')

# ===== VIDEO STREAM =====
@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ===== CURRENT PREDICTION API =====
@app.route('/get_prediction')
def get_prediction():
    return jsonify({
        'prediction': current_prediction
    })

# ===== RUN APP =====
if __name__ == "__main__":
    app.run(debug=True)