from keras import applications
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import preprocess_input

from classification.learning_model import LearningModel
from reporting.execution_parameters import BASE_PATH, DATSET_CATEGORIES_COUNT, \
  TARGET_IMAGE_DIMENSION, CURRENT_DATASET


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

  def save(self):
    pass

  def train(self):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, rotation_range=20,
        zoom_range=[0.7, 0.9],
        horizontal_flip=True,
        rescale=1. / 255)
    validation_datagen = ImageDataGenerator(
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
    validation_generator = validation_datagen.flow_from_directory(
        PATH + "_test",
        target_size=(TARGET_IMAGE_DIMENSION,
                     TARGET_IMAGE_DIMENSION),
        class_mode="categorical")
    validation_steps = validation_generator.n // validation_generator.batch_size
    best_model_path = BASE_PATH + "/models" + "/vgg_" + CURRENT_DATASET + "_model.h5"
    callbacks = [
      ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                        min_lr=0.00001, verbose=1),
      ModelCheckpoint(filepath=best_model_path, monitor='val_loss',
                      save_best_only=True, verbose=1),
    ]
    self.vgg_model.compile(loss="categorical_crossentropy",
                           optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                           metrics=["accuracy"])
    step_size_train = train_generator.n // train_generator.batch_size
    self.vgg_model.fit_generator(generator=train_generator,
                                 steps_per_epoch=step_size_train,
                                 validation_data=validation_generator,
                                 validation_steps=validation_steps,
                                 epochs=10,
                                 callbacks=callbacks)


if __name__ == '__main__':
  PATH = BASE_PATH + "/images" + "/" + CURRENT_DATASET + "_data"
  model = VggModel()
  model.prepare_for_transfer_learning()
  model.train()
  model.save()
  # list = model.get_files_list(PATH)
  # model.generate_train(list)
