'''
Base class for the various models
'''


class LearningModel:
  '''
  Adds new layers in order to support transfer learning
  '''

  def prepare_for_transfer_learning(self):
    pass

  def train(self):
    pass

  '''
  Saves the trained model to a file
  '''

  def save(self):
    pass
