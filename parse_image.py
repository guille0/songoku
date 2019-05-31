import cv2
import numpy as np
import scipy.ndimage as ndi
from helpers import crop_from_points, perspective_transform, resize_to_square
from helpers import blend_non_transparent, crop_minAreaRect, Singleton
from neural_network import NeuralNetwork
from numpy_ringbuffer import RingBuffer
import sudoku_solving
from fuzzywuzzy import fuzz, process

# For testing
from datetime import datetime


def sudoku_master(img):
    # Corners are the top left, top right, bottom left, bottom right corners
    img_processed_sudoku, corners = find_sudoku(img, draw_contours=True, test=False)

    # If we got an image
    if corners is not None:
        # We crop out the sudoku and get the info needed to paste it back
        img_cropped_sudoku, transformation_data = crop_from_points(img, corners)
        transformation_matrix = transformation_data['matrix']
        original_shape = transformation_data['original_shape']
        transformation_matrix = np.linalg.pinv(transformation_matrix)

        # We read the numbers from the cropped out sudoku
        sudoku = build_sudoku(img_cropped_sudoku, test=True)

        # The higher the threshold, the more we can approximate?
        # requires the NN to get at least a 'threshold' confidence % to consider the number as correct
        # NUMBER READING THRESHOLD
        sudoku.guess_sudoku(confidence_threshold=0.7)

        # If we read a sudoku that is 80%, 90%
        # or whatever % similar to another one we already did, we assume it's that one.
        # (it has some autocorrecting but not perfect [depends on font and neural network])
        # APPROXIMATION (modify or set it to False)
        sudoku.solve(img_cropped_sudoku, approximate=80)
        if img_cropped_sudoku is None:
            print('asdoasodads!!')

        # paste img_cropped_sudoku into img
        img_final = perspective_transform(img_cropped_sudoku, transformation_matrix, original_shape)
        img = blend_non_transparent(img, img_final)

    return img


def find_sudoku(img, draw_contours=False, test=False):
    '''Finds the biggest contoured object in the image and returns it'''
    # USED TO BE LIKE THIS: OLD
    # edges = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((3,3),np.uint8)
    # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    # edges = cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,7)
    # contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # ENDUSEDTOBE


    # Preprocessing:
    edges = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.GaussianBlur(edges,(7, 7),0)
    # edges = cv2.medianBlur(edges,5)
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges = cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,19,2)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Extracting the image of what we think might be a sudoku:
    topbottom_edges = (0, img.shape[0]-1)
    leftright_edges = (0, img.shape[1]-1)

    if len(contours)>1:

        conts = sorted(contours, key=cv2.contourArea, reverse=True)
        # *the biggest object there is*
        # Check through a few of them:

        for cnt in conts:

            epsilon = 0.025*cv2.arcLength(cnt,True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

            if len(cnt) > 3:
                topleft = min(cnt, key=lambda x: x[0,0]+x[0,1])
                bottomright = max(cnt, key=lambda x: x[0,0]+x[0,1])
                topright = max(cnt, key=lambda x: x[0,0]-x[0,1])
                bottomleft = min(cnt, key=lambda x: x[0,0]-x[0,1])
                corners = (topleft, topright, bottomleft, bottomright)

                badobject = False
                for corner in corners:
                    if corner[0][0] in leftright_edges or corner[0][1] in topbottom_edges:
                        badobject = True

                if badobject is True:
                    continue

                if test is True:
                    cv2.drawContours(img,[cnt],0,(0,255,0),2)

                # TESTING CORNERS
                # cv2.circle(img, (topleft[0][0], topleft[0][1]), 5, 0, thickness=5, lineType=8, shift=0)
                # cv2.circle(img, (topright[0][0], topright[0][1]), 5, 0, thickness=5, lineType=8, shift=0)
                # cv2.circle(img, (bottomleft[0][0], bottomleft[0][1]), 5, 0, thickness=5, lineType=8, shift=0)
                # cv2.circle(img, (bottomright[0][0], bottomright[0][1]), 5, 0, thickness=5, lineType=8, shift=0)

            else:
                return edges, None

            if cv2.contourArea(cnt) > 10000:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                if draw_contours is True:
                    cv2.drawContours(edges,[box],0,(0,255,0),2)
                
                return edges, corners

            else:
                return edges, None
    return edges, None


def build_sudoku(sudoku_image, test=False):

    # The following works very well in case i delete it:
    # edges = cv2.cvtColor(sudoku_image,cv2.COLOR_BGR2GRAY)
    # edges = cv2.dilate(edges, np.ones((2, 2)))
    # kernel = np.ones((1,5),np.uint8)
    # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    # edges = cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,5,7)

    # 'also try edges = cv2.adaptiveThreshold(edges,205,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,5,7)'

    edges = cv2.cvtColor(sudoku_image,cv2.COLOR_BGR2GRAY)
    # edges = cv2.dilate(edges, np.ones((2, 2)))
    # kernel = np.ones((1,5),np.uint8)
    # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    # for second one 7 IS GOOD
    # edges = cv2.GaussianBlur(edges,(5,5),0)
    edges = cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,5,7)

    # endtyring
    if test is True:
        cv2.imshow('edgys', edges)

    # Divide in cases and create sudoku
    h, w = sudoku_image.shape[0], sudoku_image.shape[1]

    sudoku = Sudoku.instance()
    border = 4

    for i in range(9):
        for j in range(9):
            # We get the position of each case
            x = w/9
            y = h/9

            top     = int(round(y*i+border))
            left    = int(round(x*j+border))
            right   = int(round(x*(j+1)-border))
            bottom  = int(round(y*(i+1)-border))

            point = [
                [[left,  top]],
                [[right, top]],
                [[left,  bottom]],
                [[right,  bottom]]
                ]

            # Crop out the preprocessed case (edges)
            square, _ = crop_from_points(edges, point)

            if test is True:
                if i == 8 and j == 5:
                    cv2.imshow('square', square)

            # Making the number fatter so it's easier to extract
            fat_square = square.copy()
            contours, _ = cv2.findContours(fat_square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(fat_square, contours, -1, (255,255,255), 2)


            # Extract the number (object with biggest contour)
            contours, _ = cv2.findContours(fat_square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # physical_position = [left, bottom]
            physical_position = [top, right, bottom, left]

            # TODO leave this one and remove all the others?
            # sudoku.update_case(None, (i, j), physical_position)

            if len(contours) > 0:
                conts = sorted(contours, key=cv2.contourArea, reverse=True)
                # Get the biggest one
                cnt = conts[0]

                # minarea is an arbitrary size that the number must be to be considered valid
                # TODO change it if it detects noise/doesn't detect numbers
                # minarea = w*0.3
                minarea = x*y*0.04
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
                    # if test is True:
                    #     if i == 7 and j == 2:
                    #         print(box)
                    #         print(minx, miny, maxx, maxy)
                    # Endnow

                    if number_image is None or number_image.shape[0]<2 or number_image.shape[1]<2:
                        sudoku.update_case(None, (i, j), physical_position)
                    else:
                        
                        # If we get a valid number image:
                        final = resize_to_square(number_image)

                        sudoku.update_case(final, (i, j), physical_position)
                else:
                    sudoku.update_case(None, (i, j), physical_position)
            else:
                sudoku.update_case(None, (i, j), physical_position)

    return sudoku


@Singleton
class Sudoku:
    def __init__(self):
        size = (9, 9)
        self.already_solved = {}
        self.already_solved_numbers = {}
        self.puzzle = np.empty(size, dtype=np.object)
        for i in range(size[0]):
            for j in range(size[1]):
                self.puzzle[i,j] = Case()

    
    def update_case(self, image, case_position, physical_position):
        self.puzzle[case_position].update(image, case_position, physical_position)

    def guess_sudoku(self, confidence_threshold=0):
        for i in range(9):
            for j in range(9):
                case = self.puzzle[i,j]
                case.guess_number(confidence_threshold=confidence_threshold)
                # if i == 2 and j == 0:
                #     case.print_image()
                #     cv2.imshow('square', case.image)
                #     case.write(img_cropped_sudoku)

    def write_numbers(self, sudoku_image, test=False):
        for i in range(9):
            for j in range(9):
                case = self.puzzle[i,j]
                case.write(sudoku_image) if test is False else case.testwrite(sudoku_image)
    

    def write_solution(self, sudoku_image, solution, ignore=None):
        # Turn it into an array?
        if solution is not False:
            cols   = '123456789'
            rows   = 'ABCDEFGHI'
            for i in range(9):
                for j in range(9):
                    number = solution[rows[i] + cols[j]]
                    case = self.puzzle[i,j]
                    if case.number == 0:
                        if ignore is not None and (i, j) in ignore:
                            continue
                        case.write(sudoku_image, number)


    def get_existing_numbers(self):
        existing_numbers = []
        for i in range(9):
            for j in range(9):
                case = self.puzzle[i,j]
                if case.number != 0:
                    existing_numbers.append((i,j))
        
        return existing_numbers


    def as_string(self):
        # 0:00:00.000064
        string = ''

        array = np.ravel(self.puzzle)
        for guy in array:
            string += str(guy.number)

        return string.strip()
    

    def solve_basic(self):
        string = self.as_string()
        if string in self.already_solved.keys():
            return self.already_solved[string]
        else:
            solved = sudoku_solving.solve(string)
            return solved

    
    def solve_approximate(self, approximate, test=False):
        string = self.as_string()
        if string in self.already_solved.keys():
            return self.already_solved[string], self.already_solved_numbers[string]

        else:
            # If the sudoku is unsolvable but very similar to one we already did
            # we assume it's the same one but we couldn't quite catch some numbers
            # Approximate is percent-based, 90 = 90%
            guesses = process.extract(string, self.already_solved.keys())
            for already_solved, ratio in guesses:
                if ratio > approximate:
                    if test is True:
                        print('fuzzed!', ratio)
                        print(string, 'i see')
                        print(already_solved, 'i think it is')

                    bad_solved = False

                    # We think this is it
                    # But we have to check if we accidentally saved a bad read of the sudoku

                    # WARNING TODO this is not fool proof, there could be the case
                    # where it thinks a 7 is a 1 and saves it as the correct sudoku

                    # TODO what we could do is base it on the % chance that the NN got it correct?
                    # Maybe too complicated and cpu consuming just for this
                    # Maybe just adjust the % that the NN needs to accept a number, so there's no errors.

                    for i, already_s in enumerate(already_solved):
                        # Take anything over 0
                        if already_s == '0' and string[i] != '0':
                            bad_solved = True
                        # Take 0 over 1
                        if already_s == '1' and string[i] == '0':
                            bad_solved = True

                    if bad_solved is True:
                        self.already_solved.pop(already_solved)
                        self.already_solved_numbers.pop(already_solved)
                        continue
                    # if guess is string but with some 0s instead of numbers, continue
                    return self.already_solved[already_solved], self.already_solved_numbers[already_solved]

            solved = sudoku_solving.solve(string)
            if solved is not False:
                # also save the numbers that already exist in the array
                # (so we don't write over them if we can't see them)
                self.already_solved_numbers[string] = self.get_existing_numbers()
                self.already_solved[string] = solved

                return solved, self.already_solved_numbers[string]

        return False, False


    def solve(self, img_cropped_sudoku, approximate=False):
        if approximate is False:
            solution = self.solve_basic()
            self.write_solution(img_cropped_sudoku, solution)
        
        if approximate is not False:
            solution, existing_numbers = self.solve_approximate(approximate)
            self.write_solution(img_cropped_sudoku, solution, ignore=existing_numbers)


class Case:
    def __init__(self):
        # case_position is, for example, (0,0) for the case at the top left
        # physical_position is the pixel at the center of the case
        self.image = None
        self.number = 0
        self.prev_guesses = RingBuffer(capacity=10, dtype=(float, (10)))

        self.fontsize = 0
        self.case_position = (0, 0)
        self.physical_position = (0, 0)

        self.n = 0


    def update(self, image, case_position, physical_position):
        self.image = image
        self.case_position = case_position

        top, right, bottom, left = physical_position
        average_dimension = (bottom-top + right-left)/2
        # physical_position = [left, bottom]
        # physical_position = [top, right, bottom, left]

        self.fontsize = average_dimension/40
        self.n = average_dimension/4
        # always updates physical position, even if there's no image

        self.physical_position = (physical_position[3]+1+int(self.fontsize*self.n), physical_position[2]-int(self.fontsize*self.n))


    def print_image(self):
        if self.image is not None:
            # processed = cv2.resize(self.image, (28,28), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f'number-{self.case_position[0]}-{self.case_position[1]}.jpg', self.image)


    def guess_number(self, kind=2, confidence_threshold=0):
        if kind == 1:
            if self.image is None:
                number = 0
            else:
                guy = NeuralNetwork.instance()
                prediction = guy.guess(self.image)
                number = np.argmax(prediction, axis=0)

            self.number = number
        
        if kind == 2:
            # TODO save previous guesses and average them all out

            if self.image is None:
                self.prev_guesses.appendleft(np.array([1,0,0,0,0,0,0,0,0,0]))
            else:
                guy = NeuralNetwork.instance()
                prediction = guy.guess(self.image)
                self.prev_guesses.appendleft(np.array(prediction))

            m = np.mean(self.prev_guesses, axis=0)
            number = np.argmax(m, axis=0)
            if m[number] > confidence_threshold:
                self.number = number

        return self.number


    def testwrite(self, sudoku_image):
        font = cv2.FONT_HERSHEY_TRIPLEX
        if self.image is not None:
            cv2.putText(sudoku_image, str(self.case_position[0]), self.physical_position, font, self.fontsize, (0,0,0), 1, cv2.LINE_AA)


    def write_number(self, sudoku_image):
        font = cv2.FONT_HERSHEY_TRIPLEX
        number = self.number
        if number != 0:
            cv2.putText(sudoku_image, str(number), self.physical_position, font, self.fontsize, (0,0,0), 1, cv2.LINE_AA)


    def write(self, sudoku_image, text):
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(sudoku_image, text, self.physical_position, font, self.fontsize, (0,0,0), 1, cv2.LINE_AA)


