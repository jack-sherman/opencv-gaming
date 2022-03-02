import cv2 as cv
import numpy as np
import os
# using PIL for capturing frames because it is faster than pyautogui/cv2
from PIL import ImageGrab
import time
import keyboard

start_time = time.time()

searching = True
# find the data to start the cascade classifier and initialize it
cascade_path = os.path.dirname(cv.__file__)+"/data/haarcascade_frontalface_default.xml"
faceClassifier = cv.CascadeClassifier(cascade_path)


def captureScreenRegionGRAYSCALE(box):
    # screenshot image with coordinates
    screenshot = ImageGrab.grab(bbox=box)
    # convert the screenshot to a numpy array
    screen_arr = np.array(screenshot)
    # convert the numpy array to a black and white image
    screen_image = cv.cvtColor(screen_arr, cv.COLOR_BGR2GRAY)
    return screen_image

def captureScreenRegion(box):
    # screenshot image with coordinates
    screenshot = ImageGrab.grab(bbox=box)
    # convert the screenshot to a numpy array
    screen_arr = np.array(screenshot)
    # convert the numpy array to a black and white image
    screen_image = cv.cvtColor(screen_arr, cv.COLOR_BGR2RGB)
    return screen_image

while searching:
    # pixel coordinates of image.
    # (Top left pixel x, top left pixel y, bottom right pixel x, bottom right pixel y)
    box = (300, 300, 600, 600)
    # screenshot of original image for display later
    original = captureScreenRegion(box)
    # screenshot of image in grayscale for image processing
    screen = captureScreenRegionGRAYSCALE(box)
    # use equalized histograms to improve image contrast
    screen = cv.equalizeHist(screen)
    # use the face classifier to identify all faces in a frame
    faces = faceClassifier.detectMultiScale(screen)
    # each face has a x and y coordinate along with a width and a height.
    # I'm using these to draw boxes around each face in the frame
    for (x, y, w, h) in faces:
        cv.rectangle(original, (x,y), (x+w,y+h), (0,0,255), 3)
    cv.imshow('screen', original)

    # record the framerate
    print('Frames per second: {}'.format(1 / (time.time() - start_time)))
    start_time = time.time()


    # Break on keyboard z press
    if cv.waitKey(1) == ord('z'):
        cv.destroyAllWindows()
        searching = False


