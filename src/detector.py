import cv2, numpy as np, tensorflow as tf
from .utils import preprocess_face, draw_emotion_bar, EMOTIONS
FACE_CASCADE=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

class EmotionDetector:
    def __init__(self,model_path):
        self.model=tf.keras.models.load_model(model_path)
    def detect(self,frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=FACE_CASCADE.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6,minSize=(60,60),flags=cv2.CASCADE_SCALE_IMAGE)
        for (x,y,w,h) in faces:
            face_crop=frame[y:y+h,x:x+w]
            inp=preprocess_face(face_crop)
            probs=self.model.predict(inp,verbose=0)[0]
            top=EMOTIONS[np.argmax(probs)]; conf=float(np.max(probs))
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,100),2)
            cv2.putText(frame,f'{top} ({conf:.0%})',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,100),2)
            frame=draw_emotion_bar(frame,probs,x,y,w)
        return frame
