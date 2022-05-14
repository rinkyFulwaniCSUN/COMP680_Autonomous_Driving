import collections
import cv2
import matplotlib.pyplot as plt
import numpy as np
import object_detection
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
sys.path.append('../')

# Show the version of TensorFlow and Keras that I am using
print("TensorFlow", tf.__version__)
print("Keras", keras.__version__)

def show_history(history):

  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
  plt.show()

def Transfer(n_classes, freeze_layers=True):

  print("Loading Inception V3...")


  base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

  print("Inception V3 has finished loading.")

  # Display the base network architecture
  print('Layers: ', len(base_model.layers))
  print("Shape:", base_model.output_shape[1:])
  print("Shape:", base_model.output_shape)
  print("Shape:", base_model.outputs)
  base_model.summary()


  top_model = Sequential()

  # Our classifier model will build on top of the base model
  top_model.add(base_model)
  top_model.add(GlobalAveragePooling2D())
  top_model.add(Dropout(0.5))
  top_model.add(Dense(1024, activation='relu'))
  top_model.add(BatchNormalization())
  top_model.add(Dropout(0.5))
  top_model.add(Dense(512, activation='relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(128, activation='relu'))
  top_model.add(Dense(n_classes, activation='softmax'))

  # Freeze layers in the model so that they cannot be trained (i.e. the
  # parameters in the neural network will not change)
  if freeze_layers:
    for layer in base_model.layers:
      layer.trainable = False

  return top_model


datagen = ImageDataGenerator(rotation_range=5, width_shift_range=[-10, -5, -2, 0, 2, 5, 10],
                             zoom_range=[0.7, 1.5], height_shift_range=[-10, -5, -2, 0, 2, 5, 10],
                             horizontal_flip=True)

shape = (299, 299)

# Load the cropped traffic light images from the appropriate directory
img_0_green = object_detection.load_rgb_images("traffic_light_dataset/0_green/*", shape)
img_1_yellow = object_detection.load_rgb_images("traffic_light_dataset/1_yellow/*", shape)
img_2_red = object_detection.load_rgb_images("traffic_light_dataset/2_red/*", shape)
img_3_not_traffic_light = object_detection.load_rgb_images("traffic_light_dataset/3_not/*", shape)

# Create a list of the labels that is the same length as the number of images in each

labels = [0] * len(img_0_green)
labels.extend([1] * len(img_1_yellow))
labels.extend([2] * len(img_2_red))
labels.extend([3] * len(img_3_not_traffic_light))

# Create NumPy array
labels_np = np.ndarray(shape=(len(labels), 4))
images_np = np.ndarray(shape=(len(labels), shape[0], shape[1], 3))

# Create a list of all the images in the traffic lights data set
img_all = []
img_all.extend(img_0_green)
img_all.extend(img_1_yellow)
img_all.extend(img_2_red)
img_all.extend(img_3_not_traffic_light)

# Make sure we have the same number of images as we have labels
assert len(img_all) == len(labels)

# Shuffle the images
img_all = [preprocess_input(img) for img in img_all]
(img_all, labels) = object_detection.double_shuffle(img_all, labels)

# Store images and labels in a NumPy array
for idx in range(len(labels)):
  images_np[idx] = img_all[idx]
  labels_np[idx] = labels[idx]

print("Images: ", len(img_all))
print("Labels: ", len(labels))

# Perform one-hot encoding
for idx in range(len(labels_np)):
  # We have four integer labels, representing the different colors of the
  # traffic lights.
  labels_np[idx] = np.array(to_categorical(labels[idx], 4))

# Split the data set into a training set and a validation set

idx_split = int(len(labels_np) * 0.8)
x_train = images_np[0:idx_split]
x_valid = images_np[idx_split:]
y_train = labels_np[0:idx_split]
y_valid = labels_np[idx_split:]

# Store a count of the number of traffic lights of each color
cnt = collections.Counter(labels)
print('Labels:', cnt)
n = len(labels)
print('0:', cnt[0])
print('1:', cnt[1])
print('2:', cnt[2])
print('3:', cnt[3])

# Calculate the weighting of each traffic light class
class_weight = {0: n / cnt[0], 1: n / cnt[1], 2: n / cnt[2], 3: n / cnt[3]}
print('Class weight:', class_weight)

# Save the best model as traffic.h5
checkpoint = ModelCheckpoint("trafficSinal.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(min_delta=0.0005, patience=15, verbose=1)

# Generate model using transfer learning
model = Transfer(n_classes=4, freeze_layers=True)

# Display a summary of the neural network model
model.summary()

# Generate a batch of randomly transformed images
it_train = datagen.flow(x_train, y_train, batch_size=32)

# Configure the model parameters for training
model.compile(loss=categorical_crossentropy, optimizer=Adadelta(
  lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0), metrics=['accuracy'])

history_object = model.fit(it_train, epochs=25, validation_data=(
  x_valid, y_valid), shuffle=True, callbacks=[
  checkpoint, early_stopping], class_weight=class_weight)

# Display the training history
show_history(history_object)

# Get the loss value and metrics values on the validation data set
score = model.evaluate(x_valid, y_valid, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

print('Saving the validation data set...')

print('Length of the validation data set:', len(x_valid))

# Go through the validation data set, and see how the model did on each image
for idx in range(len(x_valid)):

  # Make the image a NumPy array
  img_as_ar = np.array([x_valid[idx]])

  # Generate predictions
  prediction = model.predict(img_as_ar)

  # Determine what the label is based on the highest probability
  label = np.argmax(prediction)

  # Create the name of the directory and the file for the validation data set
  # After each run, delete this out_valid/ directory so that old files are not
  # hanging around in there.
  file_name = str(idx) + "_" + str(label) + "_" + str(np.argmax(str(y_valid[idx]))) + ".jpg"
  img = img_as_ar[0]

  # Reverse the image preprocessing process
  img = object_detection.reverse_preprocess_inception(img)

  # Save the image file
  cv2.imwrite(file_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

print('The validation data set has been saved!')
