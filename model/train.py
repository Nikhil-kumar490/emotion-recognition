"""
Train CNN for facial emotion recognition on FER-2013 dataset.
Usage: python model/train.py --data_dir data/fer2013
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 50


def build_model() -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ])
    return model


def load_fer_data(data_dir: str):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    train_gen = datagen.flow_from_directory(
        data_dir, target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale', batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        data_dir, target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale', batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation'
    )
    return train_gen, val_gen


def train(data_dir: str, output_path: str):
    train_gen, val_gen = load_fer_data(data_dir)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        callbacks.ModelCheckpoint(output_path, save_best_only=True)
    ]

    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=cb)

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss'); plt.legend()
    plt.savefig('model/training_curves.png')
    print(f"Model saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/fer2013')
    parser.add_argument('--output', default='model/emotion_model.h5')
    args = parser.parse_args()
    train(args.data_dir, args.output)
