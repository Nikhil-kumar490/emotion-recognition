# Emotion Recognition 😊😢😠

Real-time facial emotion detection using a custom CNN trained on the FER-2013 dataset, with live webcam inference via OpenCV.

## Emotions Detected
`Angry` · `Disgust` · `Fear` · `Happy` · `Sad` · `Surprise` · `Neutral`

## Tech Stack
- Python 3.10+
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib (training plots)

## Project Structure
```
emotion-recognition/
├── model/
│   ├── train.py          # CNN training script
│   ├── evaluate.py       # Evaluation & confusion matrix
│   └── emotion_model.h5  # Saved model (after training)
├── src/
│   ├── detector.py       # Face detection + emotion inference
│   ├── webcam.py         # Live webcam demo
│   └── utils.py          # Preprocessing helpers
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt

# Train model (download FER-2013 from Kaggle first)
python model/train.py --data_dir data/fer2013

# Run live webcam demo
python src/webcam.py
```

## Model Architecture
- 4x Conv2D blocks with BatchNorm + MaxPooling
- Dropout regularization (0.25, 0.5)
- Dense(256) → Dense(7, softmax)
- Input: 48×48 grayscale face crops

## Results
| Emotion | Precision | Recall |
|---------|-----------|--------|
| Happy | 0.91 | 0.93 |
| Neutral | 0.78 | 0.76 |
| Sad | 0.72 | 0.74 |
| Angry | 0.68 | 0.65 |
| Overall Accuracy | **~66%** | — |
