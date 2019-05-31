from keras.models import load_model, model_from_json
import numpy as np
import json
import cv2

from helpers import Singleton


@Singleton
class NeuralNetwork:
    def __init__(self):
        # Loads the model into memory
        with open('258epochs_model_7.json','r') as f:
            model_json = f.read()

        self.model = model_from_json(model_json)
        self.model.load_weights('258epochs_model_7.h5')
        self.labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        print('Neural Network loaded!')

    def guess(self, image):
        # Requires 28x28 images in grayscale [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]
        img_rows, img_cols = 28, 28

        image = image.reshape(1, 1, img_rows, img_cols)

        image = image.astype('float32')
        image /= 255

        prediction = self.model.predict(image)[0]

        return prediction

