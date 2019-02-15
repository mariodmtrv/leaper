import pickle

import numpy as np
from keras.models import load_model
from keras_applications.resnet50 import preprocess_input
from keras_preprocessing import image
import cv2
import keras.backend as K
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from scipy.misc import imread, imresize
from sklearn.metrics import classification_report

from reporting.execution_parameters import BASE_PATH, CURRENT_DATASET, \
  TARGET_IMAGE_DIMENSION


class ModelPredictions:
  def __init__(self):
    self.resnet_model, self.resnet_mapping = self.load_model_and_mapping(
        "resnet")
    # self.inception_model, self.inception_mapping = self.load_model_and_mapping("inception")

  def load_model_and_mapping(self, model_name):
    model = load_model(
        BASE_PATH + "/models" + "/resnet_" + CURRENT_DATASET + "_model.h5")
    # print(model.summary())
    mapping_filepath = BASE_PATH + "/models" + "/" + model_name + "_" + CURRENT_DATASET + ".mapping"
    with open(mapping_filepath, 'rb') as file:
      mapping = {v: k for k, v in pickle.load(file).items()}
    return model, mapping

  def __predict(self, model, mapping, image_subpath):
    # predicting images
    img = image.load_img(
        BASE_PATH + "/images/" + CURRENT_DATASET + "_data/" + image_subpath,
        target_size=(TARGET_IMAGE_DIMENSION, TARGET_IMAGE_DIMENSION))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    class_probabilities = model.predict(images, batch_size=10)
    best_candidate = class_probabilities.argmax(axis=-1)
    predicted_class = mapping[best_candidate[0]]
    probability = class_probabilities[0][best_candidate[0]]
    return predicted_class, probability

  def resnet_predict(self, image_subpath):
    return self.__predict(self.resnet_model, self.resnet_mapping, image_subpath)

  def generate_classification_report(self, model, mapping):
    y_true = []
    y_pred = []
    image_subpath = ""
    predicted_class,_ =  self.__predict(self, model, mapping, image_subpath)
    y_pred.append(predicted_class)
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
  predictions = ModelPredictions()
  predictions.resnet_predict('1341/44f5ae1e65bc9e43.jpg')
