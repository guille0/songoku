from __future__ import print_function
import cv2

from neural_network import NeuralNetwork
from parse_image import sudoku_master


# Capture from camera 0
cap = cv2.VideoCapture(0)
cv2.startWindowThread()

# Load neural network
NeuralNetwork.instance()

while True:
    _, img = cap.read()

    # Processes the image and outputs the image with the solved sudoku
    output = sudoku_master(img)

    cv2.imshow("input", output)

    key = cv2.waitKey(10)
    if key == 27:
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.VideoCapture(0).release()
cv2.waitKey(1)
