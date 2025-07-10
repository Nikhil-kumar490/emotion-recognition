import argparse, os, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
IMG_SIZE=48; NUM_CLASSES=7; BATCH_SIZE=64; EPOCHS=50
OUTPUT_DIR=os.path.dirname(__file__)

def build_model():
    m=models.Sequential([
        layers.Input(shape=(IMG_SIZE,IMG_SIZE,1)),
        layers.Conv2D(32,(3,3),padding='same',activation='relu'),layers.BatchNormalization(),
        layers.Conv2D(32,(3,3),padding='same',activation='relu'),layers.MaxPooling2D(2,2),layers.Dropout(0.25),
        layers.Conv2D(64,(3,3),padding='same',activation='relu'),layers.BatchNormalization(),
        layers.Conv2D(64,(3,3),padding='same',activation='relu'),layers.MaxPooling2D(2,2),layers.Dropout(0.25),
        layers.Conv2D(128,(3,3),padding='same',activation='relu'),layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),layers.Dropout(0.25),
        layers.Flatten(),layers.Dense(256,activation='relu'),layers.BatchNormalization(),layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES,activation='softmax'),
    ])
    return m

def train(data_dir,output_path):
    dg=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,rotation_range=15,horizontal_flip=True,validation_split=0.2)
    tg=dg.flow_from_directory(data_dir,target_size=(IMG_SIZE,IMG_SIZE),color_mode='grayscale',batch_size=BATCH_SIZE,class_mode='categorical',subset='training')
    vg=dg.flow_from_directory(data_dir,target_size=(IMG_SIZE,IMG_SIZE),color_mode='grayscale',batch_size=BATCH_SIZE,class_mode='categorical',subset='validation')
    model=build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='categorical_crossentropy',metrics=['accuracy'])
    cb=[callbacks.EarlyStopping(patience=10,restore_best_weights=True),callbacks.ReduceLROnPlateau(factor=0.5,patience=5),callbacks.ModelCheckpoint(output_path,save_best_only=True)]
    history=model.fit(tg,validation_data=vg,epochs=EPOCHS,callbacks=cb)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1);plt.plot(history.history['accuracy'],label='Train');plt.plot(history.history['val_accuracy'],label='Val');plt.title('Accuracy');plt.legend()
    plt.subplot(1,2,2);plt.plot(history.history['loss'],label='Train');plt.plot(history.history['val_loss'],label='Val');plt.title('Loss');plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR,'training_curves.png'))
    print(f'Model saved to {output_path}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data_dir',default='data/fer2013');p.add_argument('--output',default='model/emotion_model.h5');a=p.parse_args();train(a.data_dir,a.output)
