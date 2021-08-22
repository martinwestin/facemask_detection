# Facemask Detection
A simple program in python that is able to identify face masks in live video using a CNN.
This is a program that uses Tensorflow and OpenCV to locate face masks in live video (of course, it can also be used on still images).
It uses cascade classifier in order to locate all of the faces in an image, which then get feeded into the neural network in order
to classify if the person is wearing a facemask or not.

## Files
 - Folder "dataset" contains two folders containing images of people wearing facemasks and people not wearing face masks, used as training data.
 - Python file "data.py" is used to load data and to do preprocessing.
 - Python file "detect.py" is used to actually detect the faces in the image and show the live stream to the user.
 - Python file "model.py" is used to build and load the model.
 - Model file "model.h5" is a pretrained model. Do not edit this file manually.