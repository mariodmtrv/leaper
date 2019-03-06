import os
from enum import Enum

import pandas as pd
from sklearn.model_selection import train_test_split

from reporting.execution_parameters import RANDOM_STATE

ROOT_DIRECTORY = "/home/marioocado/Projects/university/master-thesis"
TEST_TRAIN_SPLIT_RATIO = 0.2


class Dataset:
  def __init__(self, num_categories, directory):
    self.num_categories = num_categories
    self.directory = directory


class DatasetType(Enum):
  DEV = 1,
  HYPO_EVAL = 2,
  PARAM_CONFIG = 3,
  FULL = 4,
  ML_VISION =5


'''
The various datasets used in developing the solution and their specific parameters
'''
DATASETS = {DatasetType.DEV: Dataset(10, ROOT_DIRECTORY + "/dev_data"),
            DatasetType.ML_VISION: Dataset(30,
                                           ROOT_DIRECTORY + "/ml_vision_data"),
            DatasetType.HYPO_EVAL: Dataset(100,
                                           ROOT_DIRECTORY + "/hypo_eval_data"),
            DatasetType.PARAM_CONFIG:
              Dataset(1000,
                      ROOT_DIRECTORY + "/param_config_data"),
            DatasetType.FULL: Dataset(10000, ROOT_DIRECTORY + "/full_data")}

'''
Generates a dataset with a given number of categories, ensures an 80/20 split of 
the categories between train and test data for every category. Also ensures that 
the distribution of images for each category is representative of the actual 
distribution of the whole dataset.
'''

DATASET_PATH = "/home/marioocado/Projects/university/master-thesis/" \
               "google-landmarks-dataset/train_data_cleared.csv"


class DatasetSampler:
  entire_data = pd.read_csv(DATASET_PATH)

  def __init__(self, dataset_config: DATASETS):
    self.dataset_config = dataset_config

  def select_categories(self):
    landmark_occurrences = pd.DataFrame(
        self.entire_data.landmark_id.value_counts())
    landmark_occurrences.reset_index(inplace=True)
    landmark_occurrences.columns = ['landmark_id', 'count']
    sampled = set(
        landmark_occurrences.sample(n=self.dataset_config.num_categories,
                                    random_state=RANDOM_STATE).landmark_id)

    selected_dataset = self.entire_data[
      self.entire_data.landmark_id.isin(sampled)]

    grouped = selected_dataset.groupby('landmark_id')
    train_data = pd.DataFrame(columns=selected_dataset.keys())
    test_data = pd.DataFrame(columns=selected_dataset.keys())

    for name, group in grouped:
      train_group, test_group = \
        train_test_split(group, test_size=TEST_TRAIN_SPLIT_RATIO)
      train_data = train_data.append(train_group)
      test_data = test_data.append(test_group)

    if not os.path.exists(self.dataset_config.directory):
      os.makedirs(self.dataset_config.directory)

    train_data.to_csv(self.dataset_config.directory + "/train.csv", index=False)
    test_data.to_csv(self.dataset_config.directory + "/test.csv", index=False)
    print(train_data.shape)
    print(test_data.shape)


if __name__ == '__main__':
  sampler = DatasetSampler(DATASETS[DatasetType.HYPO_EVAL])
  sampler.select_categories()
