import os
import base64
import io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

app = Flask(__name__)
# Enable CORS to prevent browser blocks if accessed via different IP aliases
CORS(app)

# Define our directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'saved_annotations')

# Ensure the folder for saved XMLs exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Load your trained AI Model here (e.g., YOLOv8)
# ==========================================================
MODEL_PATH = os.path.join(BASE_DIR, '/media/vdata/workfile/annonate_with_model/model/110.pt')  # Place your trained best.pt in this folder
model = None

print("--- AI Model Diagnostic ---")
print(f"1. Ultralytics YOLO installed: {YOLO is not None}")
print(f"2. Looking for model at: {MODEL_PATH}")
print(f"3. Model file exists: {os.path.exists(MODEL_PATH)}")

if YOLO and os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print(f"Successfully loaded AI model: {MODEL_PATH}")
else:
    print("WARNING: Model failed to load. Please fix the missing requirement above.")
print("---------------------------")

@app.route('/')
@app.route('/annotation_tool_v2.html')
def serve_tool():
    """Serve the frontend HTML tool."""
    return send_from_directory(BASE_DIR, 'annotation_tool_v2.html')

@app.route('/save_xml', methods=['POST'])
def save_xml():
    """Handle incoming XML save requests from the frontend."""
    data = request.get_json()
    
    if not data or 'filename' not in data or 'xml' not in data:
        return jsonify({"error": "Invalid payload, missing filename or xml"}), 400
    
    # Secure the filename to prevent directory traversal attacks
    filename = os.path.basename(data['filename'])
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    try:
        # Write the XML content to the file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data['xml'])
        
        print(f"Successfully saved: {filepath}")
        return jsonify({"status": "success", "file": filename}), 200
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return jsonify({"error": "Failed to save file on server"}), 500

@app.route('/get_latest_annotation', methods=['GET'])
def get_latest_annotation():
    """Fetch all saved annotation files to restore previous data."""
    try:
        # Find all xml files in the saved_annotations directory
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.xml')]
        if not files:
            return jsonify({"status": "empty", "message": "No previous data to restore."}), 200
        
        # Get latest modification time for the timestamp
        files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
        mtime = os.path.getmtime(os.path.join(OUTPUT_DIR, files[0])) # Gets timestamp of newest file
        
        annotations = []
        for f in files:
            with open(os.path.join(OUTPUT_DIR, f), 'r', encoding='utf-8') as fh:
                annotations.append({"file": f, "xml": fh.read()})
            
        return jsonify({"status": "success", "timestamp": mtime, "annotations": annotations, "img_count": len(files)}), 200
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({"error": "Failed to retrieve history"}), 500

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Receive image, run trained model inference, and return detections."""
    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({"error": "Missing image data"}), 400
        
    try:
        # 1. Extract and decode the base64 image from the frontend
        header, encoded = data['image_data'].split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        detections = [
        ]
        
        # 2. Load the image with Pillow
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 3. Run the AI Model Inference
        if model is not None:
            results = model(image)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    detections.append({
                        "label": label,
                        "x": x1, "y": y1,
                        "width": x2 - x1, "height": y2 - y1
                    })
        else:
            print("Model not loaded! Ensure 'best.pt' is in the server folder and Ultralytics is installed.")

        return jsonify({"status": "success", "detections": detections}), 200
        
    except Exception as e:
        print(f"Error during AI detection: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0 to allow other users on the network to access it
    app.run(host='0.0.0.0', port=8080, debug=True)