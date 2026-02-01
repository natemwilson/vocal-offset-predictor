# Vocal Offset Predictor

Predict word-by-word timing for subtitles to display them in sync with audio, instead of showing the entire sentence at once.

## The Problem

When watching videos with subtitles, the text often appears all at once before the audio is spoken. This creates a disconnect between reading and listening.

## The Solution

This project uses machine learning to predict how long each word takes to speak, enabling:
- Word-by-word subtitle display that follows along with the audio
- Better reading-listening synchronization
- More natural subtitle presentation

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train a Model

Using the included audiobook data:

```bash
python train.py --data "simple model using random forest regressor/Output_New.csv" --output models/duration_predictor
```

### Run Predictions

On a single sentence:
```bash
python predict.py --text "Hello, this is a test sentence." --duration 3.0 --model models/duration_predictor
```

On an SRT file:
```bash
python predict.py --input subtitles/movie.srt --model models/duration_predictor
```

### View the Demo

Run the web viewer to see word-by-word timing in action:

```bash
python run_viewer.py --model models/duration_predictor
```

Then open http://127.0.0.1:5000 in your browser.

## Project Structure

```
.
├── src/
│   ├── model/
│   │   ├── model.py      # Neural network models
│   │   ├── dataset.py    # Data loading and preprocessing
│   │   └── train.py      # Training logic
│   ├── viewer/
│   │   ├── app.py        # Flask web app
│   │   └── templates/    # HTML templates
│   └── utils/
│       └── subtitle.py   # SRT parsing and timing
├── train.py              # Training entry point
├── predict.py            # Prediction entry point
├── run_viewer.py         # Web viewer entry point
└── requirements.txt      # Dependencies
```

## Models

### SimpleDurationPredictor (Default)

A lightweight model using:
- Word embeddings
- Character-level CNN
- Bidirectional LSTM for context
- Fast inference (~1ms per sentence)

### WordDurationPredictor (Transformer)

Uses DistilBERT for more accurate predictions:
- Pre-trained language understanding
- Better handling of rare words
- Slower inference but higher accuracy

## Training Data

The model learns from audiobook data where each word's duration is known from speech recognition. The training data format:

```csv
sentence_id,word_id,word,start_time,end_time,offset
0,0,Hello,0.0,0.5,0.5
0,1,world,0.5,0.9,0.4
```

The `offset` column is the word duration in seconds.

## API

### Python

```python
from src.model import SimpleDurationPredictor

# Load model
model = SimpleDurationPredictor.load("models/duration_predictor")

# Predict timing
predictions = model.predict_sentence("Hello world")
# [('Hello', 0.32), ('world', 0.28)]
```

### Web API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "duration": 2.0}'
```

## License

MIT
