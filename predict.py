#!/usr/bin/env python3
"""
Run predictions on subtitle files.

Usage:
    python predict.py --input subtitles/movie.srt --output timed_movie.srt --model models/duration_predictor
    python predict.py --text "Hello world" --duration 2.0
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Run word duration predictions"
    )

    parser.add_argument(
        "--input", "-i",
        help="Path to input SRT file"
    )

    parser.add_argument(
        "--output", "-o",
        help="Path to output SRT file (word-by-word)"
    )

    parser.add_argument(
        "--text", "-t",
        help="Single text to predict (instead of file)"
    )

    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=5.0,
        help="Duration for single text prediction"
    )

    parser.add_argument(
        "--model", "-m",
        default="models/duration_predictor",
        help="Path to trained model"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of SRT"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference"
    )

    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
    model = None

    if model_path.exists():
        from model.model import SimpleDurationPredictor
        print(f"Loading model from {model_path}")
        model = SimpleDurationPredictor.load(str(model_path), device=args.device)
    else:
        print(f"Model not found at {model_path}, using uniform timing")

    # Single text prediction
    if args.text:
        if model:
            predictions = model.predict_sentence(args.text, device=args.device)

            # Scale to duration
            total_pred = sum(d for _, d in predictions)
            scale = args.duration / total_pred if total_pred > 0 else 1.0

            print(f"\nPredictions for: \"{args.text}\"")
            print(f"Total duration: {args.duration}s\n")
            print(f"{'Word':<20} {'Duration':<10} {'Start':<10} {'End':<10}")
            print("-" * 50)

            current = 0.0
            for word, dur in predictions:
                scaled = dur * scale
                print(f"{word:<20} {scaled:.3f}s      {current:.3f}s     {current + scaled:.3f}s")
                current += scaled
        else:
            # Uniform timing
            words = args.text.split()
            per_word = args.duration / len(words) if words else 0

            print(f"\nUniform predictions for: \"{args.text}\"")
            print(f"Total duration: {args.duration}s\n")
            print(f"{'Word':<20} {'Duration':<10} {'Start':<10} {'End':<10}")
            print("-" * 50)

            for i, word in enumerate(words):
                start = i * per_word
                end = (i + 1) * per_word
                print(f"{word:<20} {per_word:.3f}s      {start:.3f}s     {end:.3f}s")

        return

    # File prediction
    if not args.input:
        print("Error: Either --input or --text is required")
        parser.print_help()
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    from utils.subtitle import parse_srt_file, apply_word_timing, apply_uniform_timing, entries_to_json, write_timed_srt

    # Parse input
    print(f"Parsing {input_path}...")
    entries = parse_srt_file(str(input_path))
    print(f"Found {len(entries)} subtitle entries")

    # Apply timing
    if model:
        print("Applying model predictions...")
        entries = apply_word_timing(entries, model, device=args.device)
    else:
        print("Applying uniform timing...")
        entries = apply_uniform_timing(entries)

    # Output
    if args.json:
        result = entries_to_json(entries)
        output_path = args.output or input_path.with_suffix(".json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved JSON to {output_path}")
    else:
        output_path = args.output or str(input_path.with_suffix("")) + "_timed.srt"
        write_timed_srt(entries, output_path, word_by_word=True)
        print(f"Saved word-by-word SRT to {output_path}")


if __name__ == "__main__":
    main()
