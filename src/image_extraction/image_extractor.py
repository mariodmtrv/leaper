import os, urllib.request
from PIL import Image
from io import BytesIO
from reporting.execution_parameters import TARGET_IMAGE_DIMENSION

'''
Selectively downloads a single image from the given url
'''


class ImageExtractor:

  def __init__(self, out_dir, split_in_subdirs):
    self.out_dir = out_dir
    self.split_in_subdirs = split_in_subdirs

  '''
  Filters urls where download would fail
  '''

  @staticmethod
  def is_url_appropriate(url):
    # handle missing urls
    if url == 'None':
      return False
    # panoramio is a deprecated google product, will not retrieve images from there
    if 'panoramio' in url:
      return False
    return True

  '''
  Allows continuation of image downloading by preventing repetition
  '''

  @staticmethod
  def is_image_downloaded(filename):
    if os.path.exists(filename):
      print('Image %s already exists. Skipping download.' % filename)
      return True
    return False

  @staticmethod
  def download_image_data(key, url):
    try:
      response = urllib.request.urlopen(url)
      image_data = response.read()
      return image_data
    except Exception as e:
      print(str(e))
      print('Warning: Could not download image %s from %s' % (key, url))
      return None

  @staticmethod
  def parse_image_data(image_data, key):
    try:
      pil_image = Image.open(BytesIO(image_data))
      return pil_image
    except Exception as e:
      print(str(e))
      print('Warning: Failed to parse image %s' % key)
      return None

  @staticmethod
  def convert_to_rgb(pil_image, key):
    try:
      pil_image_rgb = pil_image.convert('RGB')
      return pil_image_rgb
    except Exception as e:
      print(str(e))
      print('Warning: Failed to convert image %s to RGB' % key)
      return None

  '''
  No need to store image where the size is inappropriate for data processing
  '''

  @staticmethod
  def is_image_too_small(image_data):
    return image_data.height < TARGET_IMAGE_DIMENSION \
           or image_data.width < TARGET_IMAGE_DIMENSION

  '''
  Resizes all images to processable size to save space when storing
  '''

  @staticmethod
  def resize_image(image_data):
    return image_data.resize((TARGET_IMAGE_DIMENSION, TARGET_IMAGE_DIMENSION))

  @staticmethod
  def store_image(pil_image_rgb, filename):
    try:
      pil_image_rgb.save(filename, format='JPEG', quality=90)
    except Exception as e:
      print(str(e))
      print('Warning: Failed to save image %s' % filename)
      return

  def download_image(self, key_url):
    key, url, target_dir = key_url
    constructed_dir = self.out_dir
    if self.split_in_subdirs:
      constructed_dir = os.path.join(self.out_dir, target_dir)
      if not os.path.exists(constructed_dir):
        os.mkdir(constructed_dir)

    filename = os.path.join(constructed_dir, '%s.jpg' % key)
    if self.is_image_downloaded(filename):
      return

    if not self.is_url_appropriate(url):
      return

    image_data = self.download_image_data(key, url)
    if image_data is None:
      return

    image_data_parsed = self.parse_image_data(image_data, key)
    if image_data_parsed is None:
      return

    if self.is_image_too_small(image_data_parsed):
      return

    image_resized = self.resize_image(image_data_parsed)

    pil_image_rgb = self.convert_to_rgb(image_resized, key)
    if pil_image_rgb is None:
      return

    self.store_image(pil_image_rgb, filename)
