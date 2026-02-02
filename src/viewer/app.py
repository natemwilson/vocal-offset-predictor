"""
Flask web app for viewing videos with word-by-word subtitle timing.
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

# Get the base directory
BASE_DIR = Path(__file__).parent.parent.parent


def create_app(
    model_path: str = None,
    video_dir: str = None,
    subtitle_dir: str = None
) -> Flask:
    """
    Create and configure the Flask app.

    Args:
        model_path: Path to trained model (optional)
        video_dir: Directory containing video files
        subtitle_dir: Directory containing subtitle files

    Returns:
        Configured Flask app
    """
    app = Flask(__name__, template_folder="templates")

    # Configuration
    app.config["MODEL_PATH"] = model_path
    app.config["VIDEO_DIR"] = video_dir or str(BASE_DIR / "videos")
    app.config["SUBTITLE_DIR"] = subtitle_dir or str(BASE_DIR / "subtitles")

    # Lazy load model
    _model = None

    def get_model():
        nonlocal _model
        if _model is None and app.config["MODEL_PATH"]:
            from ..model import SimpleDurationPredictor
            _model = SimpleDurationPredictor.load(app.config["MODEL_PATH"])
        return _model

    @app.route("/")
    def index():
        """Main viewer page."""
        return render_template("viewer.html")

    @app.route("/api/videos")
    def list_videos():
        """List available video files."""
        video_dir = Path(app.config["VIDEO_DIR"])
        if not video_dir.exists():
            return jsonify({"videos": []})

        videos = []
        for ext in [".mp4", ".webm", ".mkv", ".avi"]:
            videos.extend(video_dir.glob(f"*{ext}"))

        return jsonify({
            "videos": [v.name for v in sorted(videos)]
        })

    @app.route("/api/subtitles")
    def list_subtitles():
        """List available subtitle files."""
        subtitle_dir = Path(app.config["SUBTITLE_DIR"])
        if not subtitle_dir.exists():
            return jsonify({"subtitles": []})

        subtitles = list(subtitle_dir.glob("*.srt"))
        return jsonify({
            "subtitles": [s.name for s in sorted(subtitles)]
        })

    @app.route("/api/subtitle/<filename>")
    def get_subtitle(filename):
        """Get parsed subtitle with word timing."""
        from ..utils.subtitle import parse_srt_file, apply_word_timing, apply_uniform_timing, entries_to_json

        subtitle_dir = Path(app.config["SUBTITLE_DIR"])
        subtitle_path = subtitle_dir / filename

        if not subtitle_path.exists():
            return jsonify({"error": "Subtitle not found"}), 404

        entries = parse_srt_file(str(subtitle_path))

        # Apply timing based on query param
        use_model = request.args.get("model", "true").lower() == "true"

        if use_model:
            model = get_model()
            if model:
                entries = apply_word_timing(entries, model)
            else:
                entries = apply_uniform_timing(entries)
        else:
            entries = apply_uniform_timing(entries)

        return jsonify({
            "filename": filename,
            "entries": entries_to_json(entries)
        })

    @app.route("/api/predict", methods=["POST"])
    def predict():
        """Predict word timing for arbitrary text."""
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]
        duration = float(data.get("duration", 5.0))

        model = get_model()
        if not model:
            # Fallback to uniform timing
            words = text.split()
            if not words:
                return jsonify({"words": []})

            duration_per_word = duration / len(words)
            current_time = 0.0

            result = []
            for word in words:
                result.append({
                    "text": word,
                    "start_time": current_time,
                    "end_time": current_time + duration_per_word,
                    "duration": duration_per_word
                })
                current_time += duration_per_word

            return jsonify({"words": result, "source": "uniform"})

        # Use model
        word_durations = model.predict_sentence(text)

        # Scale to fit duration
        total_predicted = sum(d for _, d in word_durations)
        scale = duration / total_predicted if total_predicted > 0 else 1.0

        current_time = 0.0
        result = []

        for word, dur in word_durations:
            scaled_dur = dur * scale
            result.append({
                "text": word,
                "start_time": current_time,
                "end_time": current_time + scaled_dur,
                "duration": scaled_dur
            })
            current_time += scaled_dur

        return jsonify({"words": result, "source": "model"})

    @app.route("/videos/<path:filename>")
    def serve_video(filename):
        """Serve video files."""
        return send_from_directory(app.config["VIDEO_DIR"], filename)

    @app.route("/subtitles/<path:filename>")
    def serve_subtitle(filename):
        """Serve raw subtitle files."""
        return send_from_directory(app.config["SUBTITLE_DIR"], filename)

    return app


def run_app(
    model_path: str = None,
    video_dir: str = None,
    subtitle_dir: str = None,
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = True
):
    """
    Run the viewer app.

    Args:
        model_path: Path to trained model
        video_dir: Directory with video files
        subtitle_dir: Directory with subtitle files
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    app = create_app(model_path, video_dir, subtitle_dir)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_app()
