from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras_applications.vgg19 import preprocess_input

from classification.learning_model import LearningModel
from reporting.execution_parameters import BASE_PATH, DATSET_CATEGORIES_COUNT


class VggModel(LearningModel):
  def __init__(self):
    self.vgg_model = applications.VGG19(weights="imagenet", include_top=False,
                                        input_shape=(224, 224, 3))

  def prepare_for_transfer_learning(self):
    for layer in self.vgg_model.layers[:5]:
      layer.trainable = False
    # Adding custom Layers
    x = self.vgg_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    output = Dense(DATSET_CATEGORIES_COUNT, activation="softmax")(x)

    # creating the final model
    self.vgg_model = Model(
        self.vgg_model.input, output)

  def summary(self):
    print(self.vgg_model.summary())

  def train(self):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, rotation_range=20,
        zoom_range=[0.7, 0.9],
        horizontal_flip=True,
        rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(PATH,
                                                        target_size=(
                                                          224, 224),
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        shuffle=True)
    # compile the model
    self.vgg_model.compile(loss="categorical_crossentropy",
                           optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                           metrics=["accuracy"])
    step_size_train = train_generator.n // train_generator.batch_size
    self.vgg_model.fit_generator(generator=train_generator,
                                 steps_per_epoch=step_size_train,
                                 epochs=10)

  def save(self):
    self.vgg_model.save(MODEL_PATH)


if __name__ == '__main__':
  # print(K.tensorflow_backend._get_available_gpus())

  PATH = BASE_PATH + "/images" + "/ml_vision_data/"
  MODEL_PATH = BASE_PATH + "/models" + "/vgg_ml_vision_model.h5"
  model = VggModel()
  model.prepare_for_transfer_learning()
  model.train()
  model.save()
  # list = model.get_files_list(PATH)
  # model.generate_train(list)
