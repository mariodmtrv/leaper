import os

import numpy as np
from keras.applications import resnet50
from keras.applications.imagenet_utils import decode_predictions, \
  preprocess_input
from keras.initializers import Constant
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from reporting.execution_parameters import BASE_PATH, CURRENT_DATASET, \
  DATSET_CATEGORIES_COUNT, TARGET_IMAGE_DIMENSION

MAX_SEQUENCE_LENGTH = 16
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

class ImagenetEmbeddingsModel:
  def __init__(self):
    self.resnet_model = resnet50.ResNet50(weights='imagenet')

  def extract_image_labels(self, filepath):
    img = image.load_img(filepath, target_size=(
    TARGET_IMAGE_DIMENSION, TARGET_IMAGE_DIMENSION))
    array = image.img_to_array(img)
    expanded = np.expand_dims(array, axis=0)
    processed_image = preprocess_input(expanded)
    predictions = self.resnet_model.predict(processed_image)
    labels = decode_predictions(predictions, top=MAX_SEQUENCE_LENGTH)[0]
    extracted_text_labels = list(map(lambda x: x[1], labels))
    return ' '.join(word for word in extracted_text_labels)

  def generate_docs_and_labels(self):
    documents = []
    labels = []
    labels_index = {}
    rootDir = BASE_PATH + "/images/" + CURRENT_DATASET + "_data/"
    for fullDirName, subdirList, fileList in os.walk(rootDir, topdown=False):
      dir = fullDirName[fullDirName.rfind("/") + 1:]
      label_id = len(labels_index)
      labels_index[dir] = label_id

      print('Found directory: %s' % dir)
      for filename in fileList:
        image_subpath = rootDir + dir + '/' + filename
        current_document = self.extract_image_labels(image_subpath)
        current_label = label_id
        documents.append(current_document)
        labels.append(current_label)
    return documents, labels

  def perform(self):
    documents, labels = self.generate_docs_and_labels()
    embeddings_index = {}
    with open(os.path.join('../../resources', 'glove.6B.100d.txt')) as f:
      for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(documents)
    sequences = tokenizer.texts_to_sequences(documents)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    print("Num val samples %s" % num_validation_samples)
    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
      if i > MAX_NUM_WORDS:
        continue
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

      # load pre-trained word embeddings into an Embedding layer
      # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words, 100,
                                embeddings_initializer=Constant(
                                  embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    print('Training model.')
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    print(embedded_sequences)
    x = Conv1D(128, 3, activation='relu')(embedded_sequences)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(DATSET_CATEGORIES_COUNT, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print(model.summary())
    best_model_path = BASE_PATH + "/models" + "/embeddings_" + CURRENT_DATASET + "_model.h5"
    callbacks = [
      ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1),
      ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, verbose=1),
    ]
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              validation_data=(x_val, y_val),
              callbacks = callbacks)


if __name__ == '__main__':
  PATH = BASE_PATH + "/images" + "/" + CURRENT_DATASET + "_data"
  model = ImagenetEmbeddingsModel()
  model.perform()
