import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as mlt

mnist = tf.keras.datasets.mnist

# ------------------------ Training Data ---------------------------- #

# # x_train = handwriting & y_train = classification
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.Sequential()

# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(160, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(32, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Training
# model.fit(x_train, y_train, epochs=50)

# model.save('handwritten.model.keras')

# ----------------------------------------------------------------------- #

# ------------------------------ CLI ------------------------------------- #

model = tf.keras.models.load_model('handwritten_best.model.keras')

image_number = 1

while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The digit is probably a {np.argmax(prediction)}")
        mlt.imshow(img[0], cmap=mlt.cm.binary)
        mlt.show()
    except:
        print("Error")
    finally:
        image_number += 1

# ----------------------------------------------------------------------- #

# # --------------- Checking Accuracy and Loss ---------------------------#
# loss, accuracy = model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy)