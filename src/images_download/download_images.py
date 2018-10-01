# Landmark Recognition Challenge Image Downloader
# Originally developed by Tobias Weyand
# https://www.kaggle.com/tobwey/landmark-recognition-challenge-image-downloader
# Adapted and enhanced by Mario Dimitrov

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import os, multiprocessing, urllib.request, csv
from PIL import Image
from io import BytesIO

root_path = "/home/marioocado/Downloads/google-landmarks-dataset/"
data_file = root_path + "index.csv"
out_dir = root_path + "images"

def ParseData(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  key_url_list = [line[:2] for line in csvreader]
  return key_url_list[1:]  # Chop off header


def DownloadImage(key_url):
  (key, url) = key_url
  filename = os.path.join(out_dir, '%s.jpg' % key)

  if os.path.exists(filename):
    print('Image %s already exists. Skipping download.' % filename)
    return
  # handle missing urls
  if url == 'None':
    return
  # panoramio is a deprecated google product, will not retrieve images from there
  if 'panoramio' in url:
    return

  try:
    response = urllib.request.urlopen(url)
    image_data = response.read()
  except Exception as e:
    print(str(e))
    print('Warning: Could not download image %s from %s' % (key, url))
    return

  try:
    pil_image = Image.open(BytesIO(image_data))
  except Exception as e:
    print(str(e))
    print('Warning: Failed to parse image %s' % key)
    return

  try:
    pil_image_rgb = pil_image.convert('RGB')
  except Exception as e:
    print(str(e))
    print('Warning: Failed to convert image %s to RGB' % key)
    return

  try:
    pil_image_rgb.save(filename, format='JPEG', quality=90)
  except Exception as e:
    print(str(e))
    print('Warning: Failed to save image %s' % filename)
    return


def Run():
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  key_url_list = ParseData(data_file)
  pool = multiprocessing.Pool(processes=50)
  pool.map(DownloadImage, key_url_list)

if __name__ == '__main__':
  Run()
