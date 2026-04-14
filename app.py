import os
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB max

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi"}
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload MP4, MOV or AVI."}), 400

    # Save uploaded video
    session_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
    file.save(video_path)

    # Create output directory for this session
    output_dir = os.path.join(RESULTS_FOLDER, session_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        from stanceanalyzer import run_analysis
        results = run_analysis(video_path, output_dir, groq_api_key=GROQ_API_KEY)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        # Clean up uploaded video
        if os.path.exists(video_path):
            os.remove(video_path)

    if "error" in results:
        return jsonify(results), 400

    # Add session id so frontend can fetch images
    results["session_id"] = session_id
    return jsonify(results)


@app.route("/results/<session_id>/<filename>")
def get_result_image(session_id, filename):
    return send_from_directory(os.path.join(RESULTS_FOLDER, session_id), filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)