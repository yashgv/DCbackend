from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io
from object_detector import HomogeneousBgDetector
import logging

logging.basicConfig(filename='info.log', level=logging.DEBUG)

def log_info(req_body, res_body):
    logging.info(req_body)
    logging.info(res_body)

app = Flask(__name__)
CORS(app)

def process_image(img):
    # Load Aruco detector
    parameters = cv2.aruco.DetectorParameters()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

    # Load Object Detector
    detector = HomogeneousBgDetector()

    # Get Aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

    if len(corners) == 0:
        return img, "No ArUco marker detected"

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
        cv2.putText(img, f"Length {round(length, 2)} cm", (int(x - 140), int(y-50)), cv2.FONT_HERSHEY_PLAIN, 2.2, (100, 200, 0), 4)
        cv2.putText(img, f"Width {round(width, 2)} cm", (int(x - 140), int(y+50)), cv2.FONT_HERSHEY_PLAIN, 2.2, (100, 200, 0), 4)

    return img, "Image processed successfully"

@app.route('/process-image/', methods=['POST'])
def process_image_api():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    # Read image file
    contents = file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the image
    processed_img, message = process_image(img)

    # Encode the result image
    _, img_encoded = cv2.imencode('.png', processed_img)
    
    # Prepare the response
    response = send_file(
        io.BytesIO(img_encoded.tobytes()),
        mimetype='image/png'
    )
    response.headers["X-Process-Message"] = message
    
    return response

if __name__ == "__main__":
    app.run( debug=True)