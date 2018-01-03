# CVPyExplorer


PROGRAMMING LANGUAGE: python2
MODULES USED: tkinter, OpenCV2, sys, os, time, numpy

Demo: https://www.youtube.com/watch?v=cuirhlLaOiE

This program uses Python 2 and OpenCV 2 to attempt to detect the user's hand. Hand detected is used to interact with the interface. 

Currently works best in a monotone background with no object in the background. User may have to vary lighting to be able to detect hands properly. 

Features:

Detect a single hand and obtain contour
Determine center of the detected hand
Track the hand over multiple frames
Detect position of center of hand
If, it is location mode, this position is used to interact with file explorer interface.
Traditional mouse mode implemented(So, if user can't find ideal background they can still interact with software)
Reading text files and opening folders
Gesture Detection(moving)

Yet to be implemented:

To open files of all extensions in their respective applications
Better hand detection
Better Gesture detection

Installation:

MacOS: The program was written in MacOS and is favorable to users of this operating system. 

Mac Installation Guide
https://abhgog.gitbooks.io/open-cv-manual/content/installationenvironment-setup/mac.html

Usage:

1) Download the modules
2) Install code.py

For mouse mode:
1) User presses 's' and the file manager appears
2) The user can now interact with the buttons on the screen using mouse clicks

For location mode:
1) User first presses 's' and the file manager appears
2) The user then presses 'l' to allow the hand center to interact with the screen. 

To detect Gesture:
1) Press 'g' to start recording gesture
2) Press 'g' to stop recording gesture
If the gesture recorded exists, the code will implement the action

3) Pressing 'q' closes the window and exits program


The implementation of the code is very simple, and it can be easily converted to be able to interact with other softwares. 


Algorithm

Read from camera
Convert image to binary color using thresholding
Use OpenCV to extract contours from image
Find the largest contour from detected list, which is assumed to be the hand contour
Use OpenCV to find the convex hull and convexity defects of the contour
Find the center of the hand by using moments
Draw the convex hull, contour, center and defects on a new screen
These are then displayed on tkinter
Displays the folders at initial location stored, which is "/Users" for mac users.
Using OS module, folders inside folders are accessed
Using file handling, text files are accessed

For mouse mode, coordinates of clicks are sent and compared to location of buttons
For location mode, coordinates of center of cursor are sent and compared to location of buttons
For gesture detection, coordinates are recorded and compared
