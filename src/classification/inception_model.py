import os
import numpy as np
from keras.applications import inception_v3
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.inception_v3 import preprocess_input

from classification.learning_model import LearningModel
from image_extraction.dataset_sampler import DATASETS, DatasetType
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, \
  ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.models import load_model

from reporting.execution_parameters import BASE_PATH, CURRENT_DATASET, \
  DATSET_CATEGORIES_COUNT, TARGET_IMAGE_DIMENSION
from keras import backend as K


class InceptionModel(LearningModel):

  def __init__(self):
    self.inception_model = inception_v3.InceptionV3(weights='imagenet')

  def predict(self, files_list):
    classes = {}
    counter = 0
    for filename in files_list:
      original = load_img(PATH + filename, target_size=(TARGET_IMAGE_DIMENSION, TARGET_IMAGE_DIMENSION))
      numpy_image = img_to_array(original)
      image_batch = np.expand_dims(numpy_image, axis=0)
      # plt.imshow(np.uint8(image_batch[0]))
      processed_image = inception_v3.preprocess_input(image_batch.copy())
      predictions = self.inception_model.predict(processed_image)
      label = decode_predictions(predictions)
      cls = label[0][0][1]
      if (cls in classes):
        classes[cls] = classes[cls] + 1
      else:
        classes[cls] = 1
      counter = counter + 1
      if counter % 100 == 1:
        print(counter)
    return classes

  @staticmethod
  def get_files_list(directory):
    files_list = os.listdir(directory)
    return files_list

  def prepare_for_transfer_learning(self):
    for layer in self.inception_model.layers[:312]:
      layer.trainable = False
    self.inception_model.layers.pop()
    self.inception_model.layers[-1].outbound_nodes = []
    self.inception_model.outputs = [self.inception_model.layers[-1].output]
    output = self.inception_model.get_layer('avg_pool').output
    output = Dense(activation='relu', units=DATSET_CATEGORIES_COUNT)(output)
    output = Dense(activation='sigmoid', units=DATSET_CATEGORIES_COUNT)(
        output)
    self.inception_model = Model(self.inception_model.input, output)

    # print(self.inception_model.summary())

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
      self.inception_model.compile(optimizer='Adam',
                                   loss='categorical_crossentropy',
                                   metrics=['accuracy'])
      step_size_train = train_generator.n // train_generator.batch_size
      self.inception_model.fit_generator(generator=train_generator,
                                         steps_per_epoch=step_size_train,
                                         epochs=1)

    def save(self):
      self.inception_model.save(MODEL_PATH)


if __name__ == '__main__':
  # print(K.tensorflow_backend._get_available_gpus())

  PATH = BASE_PATH + "/images" + "/" + CURRENT_DATASET + "_data/"
  MODEL_PATH = BASE_PATH + "/models" + "/inception_" + CURRENT_DATASET + "_model.h5"
  model = InceptionModel()
  model.prepare_for_transfer_learning()
  # list = model.get_files_list(PATH)
  # model.generate_train(list)
