from keras.models import load_model

from reporting.execution_parameters import BASE_PATH


class ModelPredictions:
  def __init__(self):
    self.model = load_model(BASE_PATH + "/models" + "/inception_ml_vision_model.h5")
    print(self.model.summary())

  def predict(self):
    pass


if __name__ == '__main__':
  predictions = ModelPredictions()
