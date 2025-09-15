import argparse, cv2, sys, os
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
from src.detector import EmotionDetector

def run(model_path):
    detector=EmotionDetector(model_path)
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    if not cap.isOpened(): print('Error: Cannot open webcam.'); return
    print("Press 'q' to quit, 's' to save screenshot.")
    frame_count=0
    while True:
        ret,frame=cap.read()
        if not ret: break
        frame=detector.detect(frame)
        cv2.putText(frame,f'Frame: {frame_count}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
        cv2.imshow('Emotion Recognition',frame)
        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        if key==ord('s'): cv2.imwrite(f'screenshot_{frame_count}.png',frame)
        frame_count+=1
    cap.release(); cv2.destroyAllWindows()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',default='model/emotion_model.h5');a=p.parse_args();run(a.model)
