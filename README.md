# Emotion Recognition

Real-time facial emotion detection using a custom CNN trained on FER-2013.

## Emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## Quick Start
pip install -r requirements.txt
python model/train.py --data_dir data/fer2013
python src/webcam.py --model model/emotion_model.h5

## Docker
docker-compose up --build

## Results
| Emotion  | Precision | Recall |
|----------|-----------|--------|
| Happy    | 0.91      | 0.93   |
| Neutral  | 0.78      | 0.76   |
| Sad      | 0.72      | 0.74   |
| Angry    | 0.68      | 0.65   |
| Overall  | ~66%      |        |
