"""
Live webcam emotion recognition demo.
Usage: python src/webcam.py --model model/emotion_model.h5
"""
import argparse
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.detector import EmotionDetector


def run(model_path: str):
    detector = EmotionDetector(model_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.detect(frame)
        cv2.imshow('Emotion Recognition — Press Q to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model/emotion_model.h5')
    args = parser.parse_args()
    run(args.model)
