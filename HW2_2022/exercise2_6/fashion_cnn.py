from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from plots import plot_some_data, plot_some_predictions


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
# To do so, divide the values by 255. 
# It's important that the training set and the testing set be preprocessed in the same way
train_images = train_images / 255.0

test_images = test_images / 255.0

train_images_reshaped = train_images.reshape(-1,28,28,1)# reshape to mum_train_images X height X width X channels, where channels = 1
test_images_reshaped = test_images.reshape(-1,28,28,1)# reshape


# Build the model
# Building the neural network requires configuring the layers of the model, then compiling the model.

model = keras.Sequential()# fill the model

# 1st conv layer
model.add(layers.Conv2D(32, (3, 3), strides = 1, padding = 'same', activation = None, input_shape = (28, 28, 1)))

# 1st batch norm
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

# 2nd conv layer
model.add(layers.Conv2D(32, (3, 3), strides = 1, padding = 'same', activation = None, input_shape = (28, 28, 1)))

# 2nd batch norm
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

# 1st max pool
model.add(layers.MaxPooling2D((2, 2), padding = 'valid'))
model.add(layers.Dropout(0.2))

# 3rd conv layer
model.add(layers.Conv2D(64, (3, 3), strides = 1, padding = 'same', activation = None, input_shape = (14, 14, 32)))

# 3rd batch norm
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

# 4th conv layer
model.add(layers.Conv2D(64, (3, 3), strides = 1, padding = 'same', activation = None, input_shape = (14, 14, 64)))

# 4th batch norm
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

# 2nd max pool
model.add(layers.MaxPooling2D((2, 2), padding = 'valid'))
model.add(layers.Dropout(0.3))

# 5th conv layer
model.add(layers.Conv2D(128, (3, 3), strides = 1, padding = 'same', activation = None, input_shape = (7, 7, 64)))

# 5th batch norm
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

# 3rd max pool
model.add(layers.MaxPooling2D((2, 2), padding = 'valid'))
model.add(layers.Dropout(0.3))

#model.summary()

# flat layer
model.add(layers.Flatten())

# 1st dense layer
model.add(layers.Dense(200))

# 6th batch norm
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.Dropout(0.5))

# 2nd dense layer
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Train the model
# Training the neural network model requires the following steps:

#   1. Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
#   2. The model learns to associate images and labels.
#   3. You ask the model to make predictions about a test set—in this example, the test_images array.
#   4. Verify that the predictions match the labels from the test_labels array.

history = model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))

# summarize history for accuracy
plt.plot(history.history['accuracy'], label = 'accurancy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.title('model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train_acc', 'test_acc'], loc='lower right')
plt.show()


# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images_reshaped,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Make predictions
# With the model trained, you can use it to make predictions about some images. 
# The model's linear outputs, logits. 
# Attach a softmax layer to convert the logits to probabilities, which are easier to interpret. 
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images_reshaped)

plot_some_predictions(test_images, test_labels, predictions, class_names, num_rows=5, num_cols=3)
