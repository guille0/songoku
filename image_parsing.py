import numpy as np
import cv2

from helpers import crop_from_points, perspective_transform, resize_to_square
from helpers import blend_non_transparent
from sudoku_guessing import Sudoku


# MAIN
def sudoku_master(img_original):
    # Tries to find the part of the image with the sudoku
    ## corners are top left, top right, bottom left, bottom right
    _, corners = find_sudoku(img_original, draw_contours=False ,test=False)

    # If we got a sudoku image
    if corners is not None:
        # We crop out the sudoku and get the info needed to paste it back (matrix)
        img_cropped_sudoku, transformation = crop_from_points(img_original, corners)
        transformation_matrix = transformation['matrix']
        original_shape = transformation['original_shape']

        # We inverse the matrix so we can do the opposite transformation later
        transformation_matrix = np.linalg.pinv(transformation_matrix)

        # We crop out each number from the sudoku and create a Sudoku instance
        sudoku = build_sudoku(img_cropped_sudoku, test=True)

        # We pass the image of each case in the sudoku to a neural network to read
        ## NOTE: NUMBER READING THRESHOLD
        ## Minimum confidence the neural network needs to have about its guess (from 0 to 1)
        sudoku.guess_sudoku(confidence_threshold=0)

        # Now that we have processed the sudoku, we can solve it with a normal sudoku algorithm
        # Also writes the results into the cropped sudoku
        sudoku.solve(img_cropped_sudoku, approximate=False)
        # sudoku.write_test(img_cropped_sudoku)

        # We paste the cropped sudoku which is now solved into the camera image
        img_sudoku_final = perspective_transform(img_cropped_sudoku, transformation_matrix, original_shape)
        img_final = blend_non_transparent(img_original, img_sudoku_final)

    else:
        # If there is no sudoku to write on
        img_final = img_original

    return img_final


def find_sudoku(img, draw_contours=False, test=False):
    '''Finds the biggest object in the image and returns its 4 corners (to crop it)'''

    # Preprocessing:
    edges = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.GaussianBlur(edges, (7, 7), 0)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)

    # Get contours:
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Extracting the image of what we think might be a sudoku:
    topbottom_edges = (0, img.shape[0]-1)
    leftright_edges = (0, img.shape[1]-1)

    # NOTE change this to 0?
    # NOTE in my webcam contours[0] is always the whole image, so i just ignore it
    if len(contours) > 1:
        conts = sorted(contours, key=cv2.contourArea, reverse=True)

        # Loops through the found objects
        # for something with at least 4 corners and kinda big (>10_000 pixels)
        # NOTE change the 10000 if different webcam
        for cnt in conts:

            epsilon = 0.025*cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

            if len(cnt) > 3:
                # Gets the 4 corners of the object (assume it's a square)
                topleft =       min(cnt, key=lambda x: x[0,0]+x[0,1])
                bottomright =   max(cnt, key=lambda x: x[0,0]+x[0,1])
                topright =      max(cnt, key=lambda x: x[0,0]-x[0,1])
                bottomleft =    min(cnt, key=lambda x: x[0,0]-x[0,1])
                corners = (topleft, topright, bottomleft, bottomright)

                # Sometimes it finds 'objects' which are just parts of the screen
                # Ignore those
                badobject = False
                for corner in corners:
                    if corner[0][0] in leftright_edges or corner[0][1] in topbottom_edges:
                        badobject = True

                if badobject is True:
                    continue

                # Just a test, ignore
                if test is True:
                    cv2.drawContours(img, [cnt], 0, (0,255,0), 2)
                    # TESTING CORNERS
                    # cv2.circle(img, (topleft[0][0], topleft[0][1]), 5, 0, thickness=5, lineType=8, shift=0)
                    # cv2.circle(img, (topright[0][0], topright[0][1]), 5, 0, thickness=5, lineType=8, shift=0)
                    # cv2.circle(img, (bottomleft[0][0], bottomleft[0][1]), 5, 0, thickness=5, lineType=8, shift=0)
                    # cv2.circle(img, (bottomright[0][0], bottomright[0][1]), 5, 0, thickness=5, lineType=8, shift=0)

            else:
                # If it has less than 4 corners its not a sudoku
                return edges, None

            # NOTE edit this for different webcams, I found at least size 10k is good
            if cv2.contourArea(cnt) > 10000:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                if draw_contours is True:
                    cv2.drawContours(edges, [box], 0, (0,255,0), 2)

                # Returns the 4 corners of an object with 4+ corners and area of >10k
                return edges, corners

            else:
                return edges, None
    return edges, None


def build_sudoku(sudoku_image, test=False):
    # Different preprocessings
    # can dilate/open if numbers are small or blur if there's noise
    edges = cv2.cvtColor(sudoku_image, cv2.COLOR_BGR2GRAY)
    # edges = cv2.dilate(edges, np.ones((2, 2)))
    # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((1,5),np.uint8))
    # edges = cv2.GaussianBlur(edges,(5,5),0)
    # TODO 5,7 is normal V
    edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 7)

    # Just a test, ignore
    if test is True:
        cv2.imshow('edgys', edges)

    # Divide the sudoku in cases and load its data
    h, w = sudoku_image.shape[0], sudoku_image.shape[1]

    # Sudoku object that will contain all the information
    sudoku = Sudoku.instance()

    # NOTE Change border for different kinds of sudoku (bigger/smaller numbers, thick lines...)
    # TODO find a way to find this automatically?
    # The borders of the whole puzzle (4?)
    sudoku_border = 4
    # The lines inbetween the cases (2? 4?)
    border = 4
    x = w/9
    y = h/9

    for i in range(9):
        for j in range(9):
            # We get the position of each case (simply dividing the image in 9)


            top     = int(round(y*i+border))
            left    = int(round(x*j+border))
            right   = int(round(x*(j+1)-border))
            bottom  = int(round(y*(i+1)-border))
            if i == 0:
                top+=sudoku_border
            if i == 8:
                bottom-=sudoku_border
            if j == 0:
                left+=sudoku_border
            if j == 8:
                right-=sudoku_border

            point = [
                [[left,  top]],
                [[right, top]],
                [[left,  bottom]],
                [[right, bottom]]
                ]

            # Crop out the preprocessed case (edges)
            square, _ = crop_from_points(edges, point)

            if test is True:
                if i == 0 and j == 3:
                    cv2.imshow('square', square)
                if i == 1 and j == 0:
                    cv2.imshow('ss', square)

            # Making the number fatter so it's easier to extract
            fat_square = square.copy()
            contours, _ = cv2.findContours(fat_square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(fat_square, contours, -1, (255, 255, 255), 2)


            # Get the contour of the number (biggest object in a case)
            contours, _ = cv2.findContours(fat_square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            physical_position = [top, right, bottom, left]

            if contours:
                conts = sorted(contours, key=cv2.contourArea, reverse=True)
                # Get the biggest object in the case (assume it's a number)
                cnt = conts[0]

                # minarea is an arbitrary size that the number must be to be considered valid
                # NOTE change it if it detects noise/doesn't detect numbers (0.04)
                minarea = x*y*0.03
                if cv2.contourArea(cnt) > minarea:
                    # Crop out the number

                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    minx = max(min(box, key=lambda g: g[0])[0], 0)
                    miny = max(min(box, key=lambda g: g[1])[1], 0)
                    maxx = min(max(box, key=lambda g: g[0])[0], int(x))
                    maxy = min(max(box, key=lambda g: g[1])[1], int(y))

                    number_image = square[miny:maxy, minx:maxx]

                    if number_image is None or number_image.shape[0] < 2 or number_image.shape[1] < 2:
                        # If there's not a number in there
                        sudoku.update_case(None, (i, j), physical_position)
                    else:
                        # If we get a valid number image:
                        # Resize it to 28x28 for neural network purposes
                        final = resize_to_square(number_image)
                        # Send the data to the Sudoku object
                        sudoku.update_case(final, (i, j), physical_position)

                else:
                    sudoku.update_case(None, (i, j), physical_position)
            else:
                sudoku.update_case(None, (i, j), physical_position)

    return sudoku

