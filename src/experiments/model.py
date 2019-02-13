import keras
import numpy as np
from keras.applications import inception_v3
from keras.applications import resnet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the Inception_V3 model
inception_model = inception_v3.InceptionV3(weights='imagenet')
resnet_model = resnet50.ResNet50(weights='imagenet')

def predict(l):
  classes = {}
  counter = 0
  for filename in l:
    original = load_img(filename, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    # plt.imshow(np.uint8(image_batch[0]))
    processed_image = inception_v3.preprocess_input(image_batch.copy())
    predictions = inception_model.predict(processed_image)
    label = decode_predictions(predictions)
    cls = label[0][0][1]
    if (cls in classes):
      classes[cls] = classes[cls] + 1
    else:
      classes[cls] = 1
    counter = counter + 1
    if counter % 100 == 1:
      print(counter)
  return classes


# classes = predict(cats[0:500])
# print(classes)
# inception_model.layers
# len(inception_model.layers)
# for layer in inception_model.layers[:312]:
#   layer.trainable = False
# for layer in inception_model.layers[:313]:
#   print(layer.trainable)


model = inception_model
model.layers.pop()

model.layers[-1].outbound_nodes = []
model.outputs = [model.layers[-1].output]
output = model.get_layer('avg_pool').output
output = Dense(output_dim=10, activation='relu')(output)
output = Dense(output_dim=1, activation='sigmoid')(
  output)  # your newlayer Dense(...)
new_model = Model(model.input, output)

new_model.layers
new_model.summary()
for layer in new_model.layers[:314]:
  print(layer.trainable)

def generate_train(l):
  res = np.zeros(shape=(len(l), 224, 224, 3))
  count = 0;
  for filename in l:
    original = load_img(filename, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    #plt.imshow(np.uint8(image_batch[0]))
    #plt.show()
    processed_image = inception_v3.preprocess_input(image_batch.copy())
    res[count] = processed_image[0]
    count = count+1;
  return np.array(res);

#generate_train(cats[0:20]).shape
# X = np.concatenate((generate_train(cats[0:100]),generate_train(dogs[0:100])))
# print(X.shape)
# y = np.concatenate((np.zeros(shape=(100,1)),np.ones(shape=(100,1))))
# (y.shape)

# new_model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])


# log_file = 'output/simple_model.csv'
# best_model = 'output/simple_model.h5'
# callbacks = [
#   ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1),
#   ModelCheckpoint(filepath=best_model, monitor='val_loss', save_best_only=True, verbose=1),
#   CSVLogger(log_file)
# ]
# new_model.fit(X,y,epochs=10, batch_size=10, shuffle=True,validation_split = 0.2,callbacks=callbacks)

# log_file = 'output/simple_model.csv'
# best_model = 'output/simple_model.h5'
# model = load_model(best_model)
# predictions = model.predict_classes(X)
# predictions
# cats_test = model.predict_classes(generate_train(cats[100:200]))
# cats_test
# dogs_test = model.predict(generate_train(dogs[100:200]))
# dogs_test

model.layers[-1].outbound_nodes = []
model.outputs = [model.layers[-1].output]
output = model.get_layer('avg_pool').output
output = Dense(output_dim=1, activation='sigmoid')(output) # your newlayer Dense(...)
model2 = Model(model.input, output)

model2.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
log_file = 'output/model2.csv'
best_model = 'output/model2.h5'
callbacks = [
  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1),
  ModelCheckpoint(filepath=best_model, monitor='val_loss', save_best_only=True, verbose=1),
  CSVLogger(log_file)
]
model2.fit(X,y,epochs=10, batch_size=10, shuffle=True,validation_split = 0.2,callbacks=callbacks)


model2 = load_model(best_model)

X_test = np.concatenate((generate_train(cats[100:200]),generate_train(dogs[100:200])))
print(X_test.shape)
y_test = np.concatenate((np.zeros(shape=(100,1)),np.ones(shape=(100,1))))
print(y_test.shape)
model.evaluate(X_test,y_test)
model2.evaluate(X_test,y_test)