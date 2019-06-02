from numpy_ringbuffer import RingBuffer
from fuzzywuzzy import process
import numpy as np
import cv2

from helpers import crop_from_points, perspective_transform, resize_to_square
from helpers import blend_non_transparent, Singleton
from neural_network import NeuralNetwork
import sudoku_solving


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
        ## NOTE: APPROXIMATION (can be a % or False [default])
        ## If we read a sudoku that is very similar to one we already read,
        ## we assume it's the same one and we just couldn't see all the numbers
        ## (it has some autocorrecting but not perfect [depends on font and neural network])
        ## (it tries prioritize sudokus that actually make sense)
        sudoku.solve(img_cropped_sudoku, approximate=50)
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

    def write_solution(self, sudoku_image, solution, ignore=None):
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

    # For testing
    def write_test(self, sudoku_image):
        for i in range(9):
            for j in range(9):
                case = self.puzzle[i,j]
                if case.number != 0:
                    case.write(sudoku_image, str(case.number))

    def get_existing_numbers(self):
        existing_numbers = []
        for i in range(9):
            for j in range(9):
                case = self.puzzle[i,j]
                if case.number != 0:
                    existing_numbers.append((i,j))

        return existing_numbers

    def as_string(self):
        'Turns the numbers of the sudoku into a string to be read by algorithm'
        # 0:00:00.000064
        string = ''

        array = np.ravel(self.puzzle)
        for guy in array:
            string += str(guy.number)

        return string.strip()

    def solve_basic(self):
        'Simply reads the numbers and finds a solution. Printed numbers will be less consistent.'
        string = self.as_string()
        if string in self.already_solved.keys():
            return self.already_solved[string]
        else:
            solved = sudoku_solving.solve(string)
            return solved

    def solve_approximate(self, approximate, test=False):
        'If it finds a sudoku similar to one it has already done, uses its solution'
        string = self.as_string()
        if string in self.already_solved.keys():
            return self.already_solved[string], self.already_solved_numbers[string]

        else:
            solved = sudoku_solving.solve(string)
            # If the sudoku is unsolvable but very similar to one we already did
            # we assume it's the same one but we couldn't quite catch some numbers
            # Approximate is percent-based, 90 = 90%
            if solved is False:
                guesses = process.extract(string, self.already_solved.keys())
                for already_solved, ratio in guesses:
                    if ratio > approximate:
                        if test is True:
                            print('fuzzed!', ratio)
                            print(string, 'i see')
                            print(already_solved, 'i think it is')

                        # if guess is string but with some 0s instead of numbers, continue
                        return self.already_solved[already_solved], self.already_solved_numbers[already_solved]

            # Only saves correct solutions
            if solved is not False:
                # also save the numbers that already exist in the array
                # (so we don't write over them if we can't see them)
                self.already_solved_numbers[string] = self.get_existing_numbers()
                self.already_solved[string] = solved

                return solved, self.already_solved_numbers[string]

        return False, False


    def solve(self, img_cropped_sudoku, approximate=50):
        '''
        Approximate=False requires the NN to guess perfectly.
        Approximate=0-99 tries to solve the sudoku,
                    but if it fails it looks for a similar one that it already solved.
        '''
        if approximate is False:
            solution = self.solve_basic()
            self.write_solution(img_cropped_sudoku, solution)

        if approximate is not False:
            solution, existing_numbers = self.solve_approximate(approximate)
            self.write_solution(img_cropped_sudoku, solution, ignore=existing_numbers)


class Case:
    def __init__(self):
        # case_position is, for example, (8,8) for the case at the bottom right
        # physical_position is the pixel at the center of the case, so we know where to write
        self.image = None
        self.number = 0
        # NOTE: Edit capacity for bigger buffer?
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

        # NOTE edit this for better fontsize, positioning of the number
        self.fontsize = average_dimension/40
        self.n = average_dimension/4

        # NOTE edit this for better positioning of the number
        self.physical_position = (physical_position[3]+1+int(self.fontsize*self.n),
                                  physical_position[2]-int(self.fontsize*self.n))


    # For testing, simply saves the image of its number into a file
    def print_image(self):
        if self.image is not None:
            cv2.imwrite(f'number-{self.case_position[0]}-{self.case_position[1]}.jpg', self.image)

    def guess_number(self, kind=2, confidence_threshold=0):
        '''
        Uses neural networks to guess the number in the image.
        kind=1 is more primitive, just guesses the image (less reliable)
        kind=2 consumes more memory and CPU but is more reliable (averages out a bunch of guesses)
        '''
        if kind == 1:
            if self.image is None:
                number = 0
            else:
                guy = NeuralNetwork.instance()
                prediction = guy.guess(self.image)
                number = np.argmax(prediction, axis=0)

            self.number = number

        if kind == 2:
            # Saves a bunch of guesses (see Case.__init__ for the number)

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


    # For testing, ignore
    def testwrite(self, sudoku_image):
        font = cv2.FONT_HERSHEY_DUPLEX
        if self.image is not None:
            cv2.putText(sudoku_image, str(self.case_position[0]), self.physical_position,
                        font, self.fontsize, (0,0,0), 1, cv2.LINE_AA)


    # For testing, ignore
    def write_number(self, sudoku_image):
        font = cv2.FONT_HERSHEY_DUPLEX
        number = self.number
        if number != 0:
            cv2.putText(sudoku_image, str(number), self.physical_position,
                        font, self.fontsize, (0,0,0), 1, cv2.LINE_AA)


    def write(self, sudoku_image, text):
        'Writes the given number into the position of the case'
        # NOTE change font, colour if needed
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(sudoku_image, text, tuple(self.physical_position),
                    font, self.fontsize,(130,60,0), 1, cv2.LINE_AA)
