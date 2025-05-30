import numpy as np, cv2
EMOTIONS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
EMOTION_COLORS = {'Happy':(0,255,100),'Sad':(255,100,0),'Angry':(0,0,255),'Neutral':(200,200,200),'Fear':(255,165,0),'Surprise':(0,255,255),'Disgust':(128,0,128)}
IMG_SIZE = 48

def preprocess_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape)==3 else face_img
    return cv2.resize(gray,(IMG_SIZE,IMG_SIZE)).astype('float32')/255.0.reshape(1,IMG_SIZE,IMG_SIZE,1)

def draw_emotion_bar(frame, probs, x, y, w):
    bar_x = x + w + 10
    for i,(emotion,prob) in enumerate(zip(EMOTIONS,probs)):
        bar_y = y + i*24
        color = EMOTION_COLORS.get(emotion,(100,200,100))
        cv2.rectangle(frame,(bar_x,bar_y),(bar_x+int(prob*130),bar_y+18),color,-1)
        cv2.putText(frame,f'{emotion}: {prob:.2f}',(bar_x,bar_y+13),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)
    return frame
