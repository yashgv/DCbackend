from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Assuming HomogeneousBgDetector is in a separate file
from object_detector import *

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    print(image_path)
    # Load Aruco detector
    parameters = cv2.aruco.DetectorParameters()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

    # Load Object Detector
    detector = HomogeneousBgDetector()

    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to read the image file.")

    # Get Aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    if len(corners) == 0:
        return None, "No ArUco marker detected in the image."

    # Draw polygon around the marker
    int_corners = np.int32(corners)
    cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

    # Aruco Perimeter
    aruco_perimeter = cv2.arcLength(corners[0], True)

    # Pixel to cm ratio
    pixel_cm_ratio = aruco_perimeter / 20

    contours = detector.detect_objects(img)

    # Draw objects boundaries
    for cnt in contours:
        # Get rect
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect

        # Get Width and Height of the Objects by applying the Ratio pixel to cm
        object_width = w / pixel_cm_ratio
        object_height = h / pixel_cm_ratio

        # Determine which dimension is greater
        if object_width > object_height:
            length = object_width
            width = object_height
        else:
            length = object_height
            width = object_width

        # Display rectangle
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.polylines(img, [box], True, (255, 0, 0), 3)
        cv2.putText(img, f"Length {length:.2f} cm", (int(x - 140), int(y - 50)), cv2.FONT_HERSHEY_PLAIN, 2.2, (100, 200, 0), 4)
        cv2.putText(img, f"Width {width:.2f} cm", (int(x - 140), int(y + 50)), cv2.FONT_HERSHEY_PLAIN, 2.2, (100, 200, 0), 4)

    # Save the processed image
    filename = os.path.basename(image_path)
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + filename)
    cv2.imwrite(processed_path, img)

    return processed_path, None

@app.route('/')
def hello():
    return "Hello, World!"

@app.route('/process-image', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        processed_path, error = process_image(filepath)
        if error:
            return jsonify({'error': error}), 400

        return send_file(processed_path, mimetype='image/jpeg')

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    app.run(debug=True)