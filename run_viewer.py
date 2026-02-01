#!/usr/bin/env python3
"""
Run the subtitle viewer web app.

Usage:
    python run_viewer.py
    python run_viewer.py --model models/duration_predictor --port 8080
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from viewer.app import run_app


def main():
    parser = argparse.ArgumentParser(
        description="Run the subtitle viewer web app"
    )

    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Path to trained model (optional)"
    )

    parser.add_argument(
        "--videos", "-v",
        default="videos",
        help="Directory containing video files"
    )

    parser.add_argument(
        "--subtitles", "-s",
        default="subtitles",
        help="Directory containing subtitle files"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to"
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port to bind to"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent
    video_dir = base_dir / args.videos
    subtitle_dir = base_dir / args.subtitles

    # Create directories if they don't exist
    video_dir.mkdir(exist_ok=True)
    subtitle_dir.mkdir(exist_ok=True)

    model_path = None
    if args.model:
        model_path = str(Path(args.model))
        if not Path(model_path).exists():
            print(f"Warning: Model not found at {model_path}, running without model")
            model_path = None

    print("=" * 60)
    print("Vocal Offset Predictor - Subtitle Viewer")
    print("=" * 60)
    print(f"Model: {model_path or 'None (using uniform timing)'}")
    print(f"Videos: {video_dir}")
    print(f"Subtitles: {subtitle_dir}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)
    print("\nOpen your browser to view the subtitle timing demo")
    print("Press Ctrl+C to stop the server\n")

    run_app(
        model_path=model_path,
        video_dir=str(video_dir),
        subtitle_dir=str(subtitle_dir),
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
