import os

import numpy as np
from keras.applications import resnet50
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img, ImageDataGenerator

from classification.learning_model import LearningModel
from reporting.execution_parameters import BASE_PATH


class ResnetModel(LearningModel):
  def __init__(self):
    self.resnet_model = resnet50.ResNet50(weights='imagenet', include_top=False)

  def predict(self, files_list):
    classes = {}
    counter = 0
    for filename in files_list:
      original = load_img(PATH + filename, target_size=(224, 224))
      numpy_image = img_to_array(original)
      image_batch = np.expand_dims(numpy_image, axis=0)
      # plt.imshow(np.uint8(image_batch[0]))
      processed_image = resnet50.preprocess_input(image_batch.copy())
      predictions = self.resnet_model.predict(processed_image)
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
    output = Dense(30, activation='softmax', name='output_layer')(x)
    self.resnet_model = Model(self.resnet_model.input, output)
    self.resnet_model.summary()

    for layer in self.resnet_model.layers[:-6]:
      layer.trainable = False
    self.resnet_model.layers[-1].trainable

  def train(self):
    # print(self.resnet_model.summary())
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
    self.resnet_model.compile(optimizer='Adam', loss='categorical_crossentropy',
                              metrics=['accuracy'])
    step_size_train = train_generator.n // train_generator.batch_size
    self.resnet_model.fit_generator(generator=train_generator,
                                    steps_per_epoch=step_size_train,
                                    epochs=1)

  def save(self):
    self.resnet_model.save(MODEL_PATH)
    # self.resnet_model.predict()


if __name__ == '__main__':
  # print(K.tensorflow_backend._get_available_gpus())

  PATH = BASE_PATH + "/images" + "/ml_vision_data/"
  MODEL_PATH = BASE_PATH + "/models" + "/resnet_ml_vision_model.h5"
  model = ResnetModel()
  model.prepare_for_transfer_learning()
  # list = model.get_files_list(PATH)
  # model.generate_train(list)
