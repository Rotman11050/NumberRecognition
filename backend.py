import cv2
import numpy as np
import imageio.v2 as imageio
import tensorflow as tf
import os


class imageRecognitionAppBackend:
    def __init__(self):
            (self.train_x, self.train_y), (self.test_x, self.test_y) = tf.keras.datasets.mnist.load_data()
            (self.train_x, self.train_y), (self.test_x, self.test_y) = (self.train_x / 255.0, self.train_y), (self.test_x / 255.0, self.test_y)
            # self.model = None
    def train_model(self, epochs):
        self.model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),  
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
          ])
        self.model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

        self.model.fit(self.train_x, self.train_y, epochs=epochs)
    def predict_image(self, pil_image):
        if self.model is None:
            return None, 0.0
        
        temp_filename = "temp_image.png"
        imageio.imwrite(temp_filename, np.array(pil_image))
        img_array = imageio.imread(temp_filename)

        img_resized = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
        img_normalized = img_resized.astype("float32") / 255.0
        img_final = np.reshape(img_normalized, (1, 28, 28, 1))

        predictions = self.model.predict(img_final)
        img_normalized = img_resized.astype("float32") / 255.0
        img_final = np.reshape(img_normalized, (1, 28, 28, 1))

        prediction = self.model.predict(img_final)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        return int(predicted_digit), float(confidence)
