import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Define a model-building function
def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    
    # Tune the number of units in the first Dense layer
    hp_units1 = hp.Int('units1', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units1, activation='relu'))
    
    # Tune the number of units in the second Dense layer
    hp_units2 = hp.Int('units2', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units2, activation='relu'))
    
    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Instantiate the tuner
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

# Stop early if we find a high-accuracy model
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Perform the hyperparameter search
tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy}")

# Save the model
model.save('handwritten_best.model.keras')

# Load the best model
# model = tf.keras.models.load_model('handwritten_best.model.keras')

# Additional code for prediction with CLI can be kept the same
# ----------------------------------------------------------------------- #

# ------------------------------ CLI ------------------------------------- #

# image_number = 1

# while os.path.isfile(f"digits/digit{image_number}.png"):
#     try:
#         img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
#         img = np.invert(np.array([img]))
#         prediction = model.predict(img)
#         print(f"The digit is probably a {np.argmax(prediction)}")
#         plt.imshow(img[0], cmap=plt.cm.binary)
#         plt.show()
#     except:
#         print("Error")
#     finally:
#         image_number += 1

# ----------------------------------------------------------------------- #
