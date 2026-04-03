import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from loguru import logger
import json
import io

from src.recognition.face_recognition import FaceRecognition
from src.attendance.qdrant_db import QdrantAttendanceManager
from src.config import ApiConfig, CAPTURES_DIR
from flask import send_from_directory, Response
from src.attendance.mongodb_mgr import mongo_db

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize modules
# Note: In a production environment, you might want to use a singleton pattern or app factory
face_rec = FaceRecognition()
qdrant_mgr = QdrantAttendanceManager()

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Face Attendance API is running"}), 200

@app.route("/captures/<path:filename>")
def get_capture(filename):
    """Serve captured detection images."""
    return send_from_directory(CAPTURES_DIR, filename)

@app.route("/enroll", methods=["POST"])
def enroll_user():
    """
    Enroll a user with 3-5 images.
    Expects multipart/form-data with:
    - user_id: string (unique identifier)
    - user_name: string (display name)
    - user_info: string (optional JSON)
    - images: list of image files
    """
    try:
        user_id = request.form.get("user_id")
        user_name = request.form.get("user_name")
        user_info_raw = request.form.get("user_info")
        files = request.files.getlist("images")

        if not user_id or not user_name:
            return jsonify({"status": "error", "message": "Missing user_id or user_name"}), 400
        
        if len(files) < 3 or len(files) > 5:
            return jsonify({"status": "error", "message": "Please provide 3 to 5 images"}), 400

        try:
            user_info = json.loads(user_info_raw) if user_info_raw else {}
        except Exception:
            user_info = {"raw_info": user_info_raw}

        embeddings = []
        enrollment_image_ids = [] # To store MongoDB ObjectIds of saved images
        for file in files:
            # Read image file to numpy array
            filestr = file.read()
            nparr = np.frombuffer(filestr, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.warning(f"Could not decode image: {file.filename}")
                continue

            # Detect and extract
            faces = face_rec.detect_and_extract(img)
            if not faces:
                logger.warning(f"No face detected in {file.filename}")
                continue
            
            # Take the largest face if multiple detected
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            embeddings.append(faces[0].normed_embedding)

            # Save the enrollment image to MongoDB
            success, encoded_img = cv2.imencode('.webp', img, [int(cv2.IMWRITE_WEBP_QUALITY), 70])
            if success:
                image_id = mongo_db.save_enrollment_image(user_id, user_info.get("company_id", "default"), encoded_img.tobytes(), file.filename)
                if image_id:
                    enrollment_image_ids.append(image_id)

        if len(embeddings) < 3:
            logger.error(f"Enrollment failed for {user_name}: Only {len(embeddings)} valid faces found.")
            return jsonify({
                "status": "error", 
                "message": f"Only {len(embeddings)} faces detected. Minimum 3 required."
            }), 400

        # Save to Qdrant
        birthday = user_info.get("birthday", "N/A")
        company_id = user_info.get("company_id", "default")
        success = qdrant_mgr.upsert_user(user_name, user_id, birthday, embeddings, clear_old=True, enrollment_image_ids=enrollment_image_ids, company_id=company_id)
        
        if success:
            logger.success(f"Flask API: User '{user_name}' (ID: {user_id}) enrolled successfully.")
            return jsonify({
                "status": "success", 
                "message": f"User {user_name} enrolled successfully.",
                "user_id": user_id,
                "user_name": user_name,
                "samples": len(embeddings),
                "enrollment_image_ids": enrollment_image_ids
            }), 201
        else:
            return jsonify({"status": "error", "message": "Failed to save to database"}), 500

    except Exception as e:
        logger.error(f"Unexpected error in enroll_user: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/recognize", methods=["POST"])
def recognize_face():
    """
    Recognize a face from an uploaded image.
    Expects multipart/form-data with:
    - image: a single image file
    """
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image provided"}), 400
        
        file = request.files['image']
        filestr = file.read()
        nparr = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"status": "error", "message": "Invalid image format"}), 400

        faces = face_rec.detect_and_extract(img)
        if not faces:
            return jsonify({"status": "success", "results": [], "message": "No faces detected"}), 200

        results = []
        for face in faces:
            name, user_id, score = qdrant_mgr.recognize(face.normed_embedding)
            results.append({
                "user_id": user_id,
                "name": name,
                "confidence": float(score),
                "bbox": face.bbox.tolist()
            })

        return jsonify({
            "status": "success",
            "results": results
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/logs", methods=["GET"])
def get_logs():
    """Get list of attendance logs from MongoDB Cloud."""
    try:
        company_id = request.args.get("company_id") # Filter by company if provided
        limit = request.args.get("limit", default=100, type=int)
        
        # Fetch from MongoDB
        logs = mongo_db.get_todays_logs(company_id=company_id)
        
        # Convert MongoDB objects to Serializable dicts
        formatted_logs = []
        for log in logs:
            formatted_logs.append({
                "id": str(log["_id"]),
                "user_id": log["user_id"],
                "user_name": log["user_name"],
                "timestamp": log["timestamp"],
                "date": log["date"],
                "status": log["status"],
                "company_id": log.get("company_id"),
                "image_api_url": f"{request.host_url.rstrip('/')}/logs/{str(log['_id'])}/image"
            })
            
        return jsonify({
            "status": "success",
            "count": len(formatted_logs),
            "logs": formatted_logs[:limit]
        }), 200
    except Exception as e:
        logger.error(f"Error in get_logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/logs/<log_id>/image", methods=["GET"])
def get_log_image_api(log_id):
    """Serve the WebP image from MongoDB for a specific log."""
    try:
        image_bytes = mongo_db.get_log_image(log_id)
        if not image_bytes:
            return jsonify({"status": "error", "message": "Image not found"}), 404
        
        return Response(image_bytes, mimetype='image/webp')
    except Exception as e:
        logger.error(f"Error in get_log_image_api: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    logger.info(f"Starting Flask API on {ApiConfig.HOST}:{ApiConfig.PORT}")
    app.run(host=ApiConfig.HOST, port=ApiConfig.PORT, debug=False)
