'''
Adds features for the images
'''

import numpy as np
from keras import applications
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

from ImageMetadata import ImageMetadata


class FeatureExtraction:
  CREATED_DATE_TIME_FIELD = "DateTimeOriginal"
  FEATURE_EXTRACTION_MODEL = applications.inception_v3.InceptionV3(
      weights='imagenet', include_top=False, pooling='avg')

  def get_inception_v3_vector(self, img_path):
    x = self._preprocess_image(img_path)
    # extract the features
    features = self.FEATURE_EXTRACTION_MODEL.predict(x)[0]
    # convert from Numpy to a list of values
    features_arr = np.char.mod('%f', features)
    return features_arr

  @staticmethod
  def _preprocess_image(img_path):
    try:
      img = image.load_img(img_path, target_size=(224, 224))
      # convert image to numpy array
      x = image.img_to_array(img)
      # the image is now in an array of shape (3, 224, 224)
      # need to expand it to (1, 3, 224, 224) as it's expecting a list
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      return x
    except Exception as ex:
      print(ex)
      pass

  def get_time_of_day_created(self, exif_data):
    time = exif_data[self.CREATED_DATE_TIME_FIELD][-9:-3]
    return int(time[:3]) * 100 + int(time[4:])

  def get_exif_data(self, img_path):
    metadata_extraction = ImageMetadata(img_path)
    data = metadata_extraction.get_exif_data()
    return data

  def transform_exif_data(self, data):
    return list(data)

  def normalize_vector(self):
    pass

  def get_all_features(self, img_path):
    exif_data = self.get_exif_data(img_path)
    exif_data_values = self.transform_exif_data(exif_data.values())
    inception_vector = self.get_inception_v3_vector(img_path).tolist()
    time_created = self.get_time_of_day_created(exif_data)
    result = exif_data_values + inception_vector + [time_created]
    return result
