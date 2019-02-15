from abc import ABC, abstractmethod

'''
Base class for the various models
'''


class LearningModel(ABC):
  '''
  Adds new layers in order to support transfer learning
  '''

  @abstractmethod
  def prepare_for_transfer_learning(self):
    pass

  '''
  Uses the given training data to train the model
  '''

  @abstractmethod
  def train(self):
    pass

  '''
  Prints the summary of the model
  '''
  @abstractmethod
  def summary(self):
    pass

  '''
  Saves the trained model to a file
  '''

  @abstractmethod
  def save(self):
    pass
