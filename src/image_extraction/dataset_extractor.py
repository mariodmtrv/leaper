# Landmark Recognition Challenge Image Downloader
# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import csv
import multiprocessing
import os

from image_extraction.image_extractor import ImageExtractor
from reporting.execution_parameters import BASE_PATH


class UrlsFileParser:

  def __init__(self, data_file):
    self.data_file = data_file

  def parse_data(self):
    csvfile = open(self.data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:3] for line in csvreader]
    return key_url_list[1:]  # Chop off header


class MultiprocessDatasetExtractor:
  def __init__(self, selected_dir, split_in_dirs):
    self.split_in_subdirs = split_in_dirs
    self.ROOT_PATH = BASE_PATH + selected_dir
    self.DATA_FILE = self.ROOT_PATH + "/train.csv"
    self.OUT_DIR = BASE_PATH + "/images" + selected_dir

  def extract_images(self):
    if not os.path.exists(self.OUT_DIR):
      os.mkdir(self.OUT_DIR)
    file_parser = UrlsFileParser(self.DATA_FILE)
    key_url_list = file_parser.parse_data()
    image_extractor = ImageExtractor(self.OUT_DIR, self.split_in_subdirs)
    pool = multiprocessing.Pool(processes=50)
    pool.map(image_extractor.download_image, key_url_list)


if __name__ == '__main__':
  dataset_extractor = MultiprocessDatasetExtractor("/dev_data", False)
  dataset_extractor.extract_images()
