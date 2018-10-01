import unittest

from FeatureExtraction import FeatureExtraction


class FeatureExtractionTest(unittest.TestCase):
  path = '../resources/source_images/DSC00312.JPG'

  def test_exif_extraction(self):
    extraction = FeatureExtraction()
    exif_data = extraction.get_exif_data(self.path)
    # print(exif_data.keys())
    time_created = extraction.get_time_of_day_created(exif_data)
    self.assertEqual(time_created, 1443)

  def test_image_features_extraction(self):
    extraction = FeatureExtraction()
    features = extraction.get_all_features(self.path)
    self.assertEqual(len(features), 2099)
    # print(features)


if __name__ == '__main__':
  unittest.main()
