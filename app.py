import os
import uuid
import json
import queue
import threading
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi"}
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Store results keyed by session_id
analysis_results = {}
analysis_queues = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded."}), 400
    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Please upload MP4, MOV or AVI."}), 400

    session_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
    file.save(video_path)

    speed = float(request.form.get("speed", 1.0))

    output_dir = os.path.join(RESULTS_FOLDER, session_id)
    os.makedirs(output_dir, exist_ok=True)

    q = queue.Queue()
    analysis_queues[session_id] = q

    def run():
        from stanceanalyzer import run_analysis

        def progress_callback(msg):
            q.put({"type": "progress", "data": msg})

        result = run_analysis(
            video_path, output_dir,
            groq_api_key=GROQ_API_KEY,
            progress_callback=progress_callback,
            speed=speed
        )

        if os.path.exists(video_path):
            os.remove(video_path)

        analysis_results[session_id] = result
        q.put({"type": "done", "session_id": session_id})

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return jsonify({"session_id": session_id})


@app.route("/stream/<session_id>")
def stream(session_id):
    def generate():
        q = analysis_queues.get(session_id)
        if not q:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
            return

        while True:
            try:
                msg = q.get(timeout=120)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Timeout'})}\n\n"
                break

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/results_data/<session_id>")
def results_data(session_id):
    result = analysis_results.get(session_id)
    if not result:
        return jsonify({"error": "Results not found."}), 404
    return jsonify(result)


@app.route("/results/<session_id>/<filename>")
def get_result_image(session_id, filename):
    return send_from_directory(os.path.join(RESULTS_FOLDER, session_id), filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)