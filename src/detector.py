import cv2
import numpy as np
import tensorflow as tf
from .utils import preprocess_face, draw_emotion_bar, EMOTIONS

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class EmotionDetector:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect faces and overlay emotion predictions on the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            input_tensor = preprocess_face(face_crop)
            probs = self.model.predict(input_tensor, verbose=0)[0]
            top_emotion = EMOTIONS[np.argmax(probs)]
            confidence = float(np.max(probs))

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 100), 2)
            label = f'{top_emotion} ({confidence:.0%})'
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

            # Draw probability bars
            frame = draw_emotion_bar(frame, probs, x, y, w)

        return frame
