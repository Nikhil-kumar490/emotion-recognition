import numpy as np
import cv2

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = 48


def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """Resize, normalize, and reshape a face crop for model input."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype('float32') / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def draw_emotion_bar(frame: np.ndarray, probs: np.ndarray, x: int, y: int, w: int):
    """Draw a probability bar chart overlay on the frame."""
    bar_x = x + w + 10
    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        bar_y = y + i * 22
        bar_len = int(prob * 120)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 16), (100, 200, 100), -1)
        cv2.putText(frame, f'{emotion}: {prob:.2f}', (bar_x, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return frame
