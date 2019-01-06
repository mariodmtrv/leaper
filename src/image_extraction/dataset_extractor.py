# Landmark Recognition Challenge Image Downloader
# Inspired by Tobias Weyand
# https://www.kaggle.com/tobwey/landmark-recognition-challenge-image-downloader

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import os, multiprocessing, csv

from image_extraction.image_extractor import ImageExtractor

BASE_PATH = "/home/marioocado/Projects/university/master-thesis"
ROOT_PATH = BASE_PATH + "/ml_vision_data"
DATA_FILE = ROOT_PATH + "/train.csv"
OUT_DIR = BASE_PATH + "/images/ml_vision_data"


class UrlsFileParser:

  def __init__(self, data_file):
    self.data_file = data_file

  def parse_data(self):
    csvfile = open(self.data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:3] for line in csvreader]
    return key_url_list[1:]  # Chop off header


class MultiprocessDatasetExtractor:
  def __init__(self, split_in_dirs):
    self.split_in_subdirs = split_in_dirs

  def extract_images(self):
    if not os.path.exists(OUT_DIR):
      os.mkdir(OUT_DIR)
    file_parser = UrlsFileParser(DATA_FILE)
    key_url_list = file_parser.parse_data()
    image_extractor = ImageExtractor(OUT_DIR, self.split_in_subdirs)
    pool = multiprocessing.Pool(processes=50)
    pool.map(image_extractor.download_image, key_url_list)


if __name__ == '__main__':
  dataset_extractor = MultiprocessDatasetExtractor(True)
  dataset_extractor.extract_images()
