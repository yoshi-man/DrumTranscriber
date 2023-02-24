import os

import numpy as np

from tensorflow.keras import models, layers, optimizers, utils
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocessing import Dataset, Preprocessor

from utils.config import SETTINGS
from datetime import datetime

import mlflow
import mlflow.keras

MODEL_PATH = None


def get_model(path=None):
    if path is None:
        conv_base = InceptionResNetV2(weights="imagenet",
                                      include_top=False,
                                      input_shape=(256, 256, 3)
                                      )

        # make it so the conv_base is not trainable
        conv_base.trainable = False

        # add more layers on top of the Inception model
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(2048, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.15))
        model.add(layers.Dense(6, activation='sigmoid'))  # 6 classes

    else:
        model = models.load_model(path)

    return model


if __name__ == '__main__':
    mlflow.tensorflow.autolog()

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        './dataset/train',
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical')
    validation_generator = val_datagen.flow_from_directory(
        './dataset/val',
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(
        './dataset/test',
        target_size=(256, 256),
        batch_size=64,
        class_mode='categorical')

    # initialise and build CNN model based on InceptionResNetV2
    model = get_model(MODEL_PATH)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.0005),
                  metrics=['acc'])

    history = model.fit(
        train_generator,
        steps_per_epoch=64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=16,
        shuffle=True)

    model.evaluate(test_generator)
