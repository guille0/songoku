from keras.models import load_model
import numpy as np
import cv2
from helpers import Singleton

from keras.models import model_from_json
import json


@Singleton
class NeuralNetwork:
    def __init__(self):
        with open('258epochs_model_7.json','r') as f:
            model_json = f.read()

        self.model = model_from_json(model_json)
        self.model.load_weights('258epochs_model_7.h5')
        # self.model = load_model('258epochs_model_7.h5')
        self.labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        print('Neural Network loaded!')

    def guess(self, image):
        img_rows, img_cols = 28, 28
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, 1, img_rows, img_cols)

        image = image.astype('float32')
        image /= 255

        prediction = self.model.predict(image)[0]

        # best = np.argmax(prediction)

        # result = self.labels[best]

        return prediction


# guy = NeuralNetwork.instance()
# print('finished loading')
# image = cv2.imread('fonts/1-2.png')
# # image2 = cv2.imread('img_1.jpg')
# print(guy.guess(image))
# # print(guy.guess(image2))