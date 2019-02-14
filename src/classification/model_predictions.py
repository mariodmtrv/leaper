import numpy as np
from keras.models import load_model
from keras_preprocessing import image
import cv2
import keras.backend as K
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

from reporting.execution_parameters import BASE_PATH, CURRENT_DATASET, \
  TARGET_IMAGE_DIMENSION


class ModelPredictions:
  def __init__(self):
    self.model = load_model(BASE_PATH + "/models" + "/resnet_"+ CURRENT_DATASET+ "model.h5")
    #print(self.model.summary())

  def predict(self, image_subpath):
    # predicting images
    img = image.load_img(BASE_PATH+ "/images/"+ CURRENT_DATASET+ "_data/" + image_subpath, target_size=(TARGET_IMAGE_DIMENSION, TARGET_IMAGE_DIMENSION))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    y_proba = self.model.predict(images, batch_size=10)
    print(y_proba)




if __name__ == '__main__':
  predictions = ModelPredictions()
  predictions.heatmap_for_image('1341/44f5ae1e65bc9e43.jpg')

