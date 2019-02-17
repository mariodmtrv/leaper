import os

import numpy as np
from keras.applications import inception_v3
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img, ImageDataGenerator

from classification.learning_model import LearningModel
from reporting.execution_parameters import BASE_PATH, CURRENT_DATASET, \
  DATSET_CATEGORIES_COUNT, TARGET_IMAGE_DIMENSION


class InceptionModel(LearningModel):
  def __init__(self):
    self.inception_model = inception_v3.InceptionV3(weights='imagenet',
                                                    include_top=False)

  @staticmethod
  def get_files_list(directory):
    files_list = os.listdir(directory)
    return files_list

  def prepare_for_transfer_learning(self):
    # for layer in self.inception_model.layers[:312]:
    #   layer.trainable = False
    # self.inception_model.layers.pop()
    # self.inception_model.layers[-1].outbound_nodes = []
    # self.inception_model.outputs = [self.inception_model.layers[-1].output]
    # output = self.inception_model.get_layer('avg_pool').output
    # output = Dense(activation='relu', units=DATSET_CATEGORIES_COUNT)(output)
    # output = Dense(activation='softmax', units=DATSET_CATEGORIES_COUNT)(
    #     output)
    for layer in self.inception_model.layers[:]:
      layer.trainable = False
    x = self.inception_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(DATSET_CATEGORIES_COUNT, activation='softmax')(x)
    self.inception_model = Model(self.inception_model.input, output)

  def summary(self):
    print(self.inception_model.summary())

  def train(self):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, rotation_range=20,
        zoom_range=[0.7, 0.9],
        horizontal_flip=True,
        rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(PATH,
                                                        # this is where you specify the path to the main data folder
                                                        target_size=(
                                                          224, 224),
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        shuffle=True)
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, rotation_range=20,
        zoom_range=[0.7, 0.9],
        horizontal_flip=True,
        rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        PATH + "_test",
        target_size=(TARGET_IMAGE_DIMENSION,
                     TARGET_IMAGE_DIMENSION),
        class_mode="categorical")
    validation_steps = validation_generator.n // validation_generator.batch_size
    best_model_path = BASE_PATH + "/models" + "/inception_" + CURRENT_DATASET + "_model.h5"

    callbacks = [
      ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                        min_lr=0.00001, verbose=1),
      ModelCheckpoint(filepath=best_model_path, monitor='val_loss',
                      save_best_only=True, verbose=1),
    ]
    self.inception_model.compile(optimizer='Adam',
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])
    step_size_train = train_generator.n // train_generator.batch_size
    self.inception_model.fit_generator(generator=train_generator,
                                       steps_per_epoch=step_size_train,
                                       validation_data=validation_generator,
                                       validation_steps=validation_steps,
                                       epochs=5,
                                       callbacks=callbacks)

  def save(self):
    self.inception_model.save(MODEL_PATH)


if __name__ == '__main__':
  PATH = BASE_PATH + "/images" + "/" + CURRENT_DATASET + "_data"
  MODEL_PATH = BASE_PATH + "/models" + "/inception_" + CURRENT_DATASET + "_model.h5"
  model = InceptionModel()
  model.prepare_for_transfer_learning()
  model.summary()
  model.train()
