import os
import pickle

import cv2
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import resnet50
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imread, imresize

from classification.learning_model import LearningModel
from reporting.execution_parameters import BASE_PATH, CURRENT_DATASET, \
  DATSET_CATEGORIES_COUNT, TARGET_IMAGE_DIMENSION


class ResnetModel(LearningModel):
  def __init__(self):
    self.resnet_model = resnet50.ResNet50(weights='imagenet', include_top=False)
    self.is_trained = False

  def get_files_list(self, directory):
    files_list = os.listdir(directory)
    return files_list

  def prepare_for_transfer_learning(self):
    last_layer = self.resnet_model.output
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(last_layer)
    # add fully-connected & dropout layers
    x = Dense(512, activation='relu', name='fc-1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='fc-2')(x)
    x = Dropout(0.5)(x)
    # a softmax layer for 4 classes
    output = Dense(DATSET_CATEGORIES_COUNT, activation='softmax',
                   name='output_layer')(x)
    self.resnet_model = Model(self.resnet_model.input, output)
    self.resnet_model.summary()

    for layer in self.resnet_model.layers[:-6]:
      layer.trainable = False
    self.resnet_model.layers[-1].trainable

  def summary(self):
    print(self.resnet_model.summary())

  def train(self):
    # print(self.resnet_model.summary())
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, rotation_range=20,
        zoom_range=[0.7, 0.9],
        horizontal_flip=True,
        rescale=1. / 255)
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, rotation_range=20,
        zoom_range=[0.7, 0.9],
        horizontal_flip=True,
        rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(PATH,
                                                        target_size=(
                                                          TARGET_IMAGE_DIMENSION,
                                                          TARGET_IMAGE_DIMENSION),
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        shuffle=True)
    validation_generator = validation_datagen.flow_from_directory(
        PATH + "_test",
        target_size=(TARGET_IMAGE_DIMENSION,
                     TARGET_IMAGE_DIMENSION),
        class_mode="categorical")

    self.store_class_mapping(train_generator)
    self.resnet_model.compile(optimizer='Adam', loss='categorical_crossentropy',
                              metrics=['accuracy'])

    step_size_train = train_generator.n // train_generator.batch_size
    validation_steps = validation_generator.n // validation_generator.batch_size
    best_model_path = BASE_PATH + "/models" + "/resnet_" + CURRENT_DATASET + "_model.h5"

    callbacks = [
      ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1),
      ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, verbose=1),
    ]
    self.resnet_model.fit_generator(generator=train_generator,
                                    steps_per_epoch=step_size_train,
                                    validation_data=validation_generator,
                                    validation_steps= validation_steps,
                                    epochs=5,
                                    callbacks = callbacks)
    self.is_trained = True

  def store_class_mapping(self, train_generator):
    filepath = BASE_PATH + "/models" + "/resnet_" + CURRENT_DATASET + "_2.mapping"
    with open(filepath, 'wb') as file:
      label_map = (train_generator.class_indices)
      pickle.dump(label_map, file, pickle.HIGHEST_PROTOCOL)

  def save(self):
    self.resnet_model.save(MODEL_PATH)

  def get_image(self, image_fname):
    original = imread(image_fname)
    image = imresize(original, (TARGET_IMAGE_DIMENSION, TARGET_IMAGE_DIMENSION))
    image = np.expand_dims(image, axis=0)
    return image, original

  """
  Generate heatmap to show activation for the last conv layer
  Sanity check that the model learns roughly what a human would use to identify the image
  """

  def heatmap_for_image(self, image_subpath):
    """
    Load the model if it is not available
    """
    if not self.is_trained:
      self.resnet_model = load_model(
          BASE_PATH + "/models" + "/resnet_" + CURRENT_DATASET + "_model.h5")
      self.is_trained = True
    conv_layer = self.resnet_model.layers[-7]
    gradients = K.gradients(self.resnet_model.output, conv_layer.output)[0]
    mean_gradients = K.mean(gradients, axis=(0, 1, 2))
    image, original = self.get_image(
        BASE_PATH + "/images/" + CURRENT_DATASET + "_data/" + image_subpath)
    extractor_fn = K.function([self.resnet_model.input],
                              [mean_gradients, conv_layer.output[0]])
    mean_grad_vals, conv_output = extractor_fn([image])
    num_classes = self.resnet_model.layers[-1].output.shape[1]
    for i in range(num_classes):
      conv_output[:, :, i] *= mean_grad_vals[i]
    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    upsized_cam_image = cv2.resize(heatmap, original.shape[:2][::-1])
    # overlay with original image
    heatmap = cv2.applyColorMap(np.uint8(255 * upsized_cam_image),
                                cv2.COLORMAP_JET)
    plt.imshow(original)
    plt.axis('off')
    # plt.imshow(heatmap, cmap='cool', alpha=0.4)
    plt.imshow(heatmap, alpha=0.4)
    plt.show()


if __name__ == '__main__':
  PATH = BASE_PATH + "/images" + "/" + CURRENT_DATASET + "_data"
  MODEL_PATH = BASE_PATH + "/models" + "/resnet_" + CURRENT_DATASET + "_model.h5"
  model = ResnetModel()
  # model.prepare_for_transfer_learning()
  # model.train()
  model.heatmap_for_image('326/00bc25c5f45a53a6.jpg')
