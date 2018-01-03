#################################################

# TERM PROJECT - AirPynteract
# A 112 term project by Shivani Prasad
# andrewid: sprasad1
# Term: Fall 2017

#################################################
#DOCUMENTATION:
#Template for displaying cv2 output on tkinter taken from:
#       https://github.com/VasuAgrawal/112-opencv-tutorial/blob/master/
#                                              opencvTkinterTemplate.py

#Modules required

import time
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")

# Tkinter selector
if sys.version_info[0] < 3:
    from Tkinter import *
    import Tkinter as tk
else:
    from tkinter import *
    import tkinter as tk

import numpy as np
import cv2
from PIL import Image, ImageTk
import copy
import math
import os

#################################################

# GLOBAL VARIABLES
# (Globally defined variables that remain static)

#1) Colors
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
yellow = (0, 255, 255)
white = (255, 255, 255)

#2) List of Moments
x0 = [0]
y0 = [0]

#global variables used
def globalVariables(data, x0, y0):
    data.n = 0
    data.color = ["#D4D3D0"]
    data.cLen = len(data.color)
    data.ratio = 920/350  #values taken from frame size 1080x1920
    data.firstIter = True
    data.x0 = x0
    data.y0 = y0   
    
    data.initPath = "/Users"
    data.currPath = data.initPath
    data.directory = os.listdir(data.currPath)
    data.directory = filterDirectories(data.directory)
    data.directory = createInitialVisibility(data)
    data.move = None #swipe up or down
    data.radius = 10
    
    data.firstIteration = True
    data.button = []
    
    data.locationMode = False
    data.l = False
    data.fileMode = False
    
    data.tkinterCenter = (0,0)
    
    data.time = 0
    
    data.lines = []
    data.textStart = 0
    data.startScreen = True
    
    data.gesture = None
    data.gStart = []
    data.gEnd = []


# Helper Functions 

# FRAME MANIPULAITON:
# (Functions required to capture data from the webcam)

#1)  Converts image to grayscale
# NOTE: webcams usually capture image in RGB/BGR format
def toGrayscale(input):
    return cv2.cvtColor(input,cv2.COLOR_BGR2GRAY) 

#2) Blurs the image
def toBlur(input):
    blur = cv2.GaussianBlur(input,(41,41),0)
    blur = cv2.bilateralFilter(blur, 10, 50, 100) 
    return blur

#3) Converts image to binary threshold 
def toThreshold(input):
    _, thresh = cv2.threshold(input,70,255,
                cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
    return thresh


# FEATURE DETECTION:
# (Functions required to detect features of the hand contour)

#1) Extracts contours from the given frame/input
def extractContours(input):
    _, contours, _ = cv2.findContours(input, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours
    
#2) Extracts contour of hand
def extractHandContour(contours):
    maxArea, index = -1, -1
    for i in range (len(contours)):
        area = cv2.contourArea(contours[i])
        if area > maxArea:
            maxArea = area
            index = i
    return contours[index]
    
#3) Extracts length of contour
def extractContourLength(contour):
    return cv2.arcLength(contour, True)
    
#4) Extracts convex hull of hand contour
def extractConvexHull(contour):
    hull1 = cv2.convexHull(contour)
    hull2 = cv2.convexHull(contour, returnPoints = False)
    return hull1, hull2
    
#5) Extracts reduced contour
def extractReducedContour(contour, arc):
    return cv2.approxPolyDP(contour, arc/1000, True)
    
#6) Extracts hull points
def extractHullPoints(contour, hull):
    hullPoints = [contour[i[0]] for i in hull]
    return np.array(hullPoints, dtype=np.int32)
    
#7) Extracts hull defects
def extractDefects(contour, hull):
    return cv2.convexityDefects(contour, hull)

#8) Extract Centre of Hand
#DOCUMENTATION: 
#           https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
def extractCentre(contour, x0, y0):
    M = cv2.moments(contour)
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except:
        print("One of the moments was 0 so center could not be calculated")
        cX = x0[-1]
        cY = y0[-1]
    x0 += [cX]
    y0 += [cY]
    return tuple([cX, cY])
    
# DRAWING
# (Functions required to draw on the screen)

#1) Draws contour
def drawContour(contour, output):
    cv2.drawContours(output, [contour],-1, green, 2)

#2) Draws convex hull
def drawHull(hull, output):
    cv2.drawContours(output, [hull], 0, red, 2)
    
#3) Draws defect points
# Documentation:
#           http://opencvpython.blogspot.com/2012/06/contours-4-ultimate.html
def drawDefects(defects, contour, output):
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        cv2.circle(output,far,3,yellow,-1)
        cv2. circle(output, far, 7, yellow, 2)
        
#4) Draws center of contour
def drawCenter(center, output):
    (cX, cY) = center
    cv2.circle(output, center, 5, white, -1)
    cv2.putText(output, "center", (cX - 20, cY - 20),\
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1)
    
#creates initial list of files
def createInitialVisibility(data):
    p = []
    for x in range (len(data.directory)):
        if x < 3:
            p.append([data.directory[x], 1])
        else:
            p.append([data.directory[x], 0])
    return p

#filters directory by getting rid of hidden files the user should not be able to access    
def filterDirectories(directory):
    y = []
    for x in directory:
        if x[0] == '.':
            continue
        else: y.append(x)
    y.sort()
    return y 

#draws screen
def createScreen(canvas, data):
    x = 0
    init = 190
    end = 415
    gap = 75
    index = 0
    
    if data.firstIteration ==True: 
        data.button = []
            
    while (x != 3 and index < len(data.directory)):
        if data.directory[index][1] == 1:
            if data.firstIteration == True:
                data.button.append([data.directory[index][0], 400+data.radius, init+x*gap+data.radius])
            canvas.create_text(440, init + x*gap, text = data.directory[index][0],\
            anchor = "nw")
            canvas.create_oval(400, init+x*gap, 400+2*data.radius,init+x*gap+2*data.radius, fill = "#6699cc", width = 0)
            x += 1
        index +=1
    
    data.firstIteration = False

def distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    
def drawScreen(canvas, data):
    canvas.create_rectangle(0,0, data.width, data.height, fill= "#686868", outline =  data.color[data.n%data.cLen], width =5)
    
    canvas.create_rectangle(30,30,data.width-30, 550, fill = data.color[data.n%data.cLen],outline = "", width = 0)
    
    drawCamera(canvas, data)
    
    canvas.create_line(0,0,0,data.height, width = 15, fill = data.color[data.n%data.cLen])
    canvas.create_line(0,0,data.width,0, width = 15, fill = data.color[data.n%data.cLen])
    canvas.create_line(0,data.height,data.width,data.height, width = 15, fill = data.color[data.n%data.cLen])
    
    canvas.create_rectangle(150, 40, 1050, 540, fill = "white", width = 2, outline = "#C0C0C0")
    canvas.create_rectangle(150, 40, 1050, 90, fill = "#C0C0C0", width = 0)
    canvas.create_rectangle(150, 450, 1050, 540, fill = "#C0C0C0", width = 0)
    
    canvas.create_rectangle(575, 115, 625, 140, fill = "#888888", width = 0)
    canvas.create_text(600, 127, text = "Up",font ="Helvetica 16 bold", fill = "white")
    canvas.create_rectangle(575, 410, 625, 435, fill = "#888888", width = 0)
    canvas.create_text(600, 423, text = "Down",font ="Helvetica 16 bold", fill = "white")
    
    canvas.create_rectangle(380, 112, 460, 143, fill = "#fb5d5d", width = 0 )
    canvas.create_text( 420, 127, text = "Go Back", font ="Helvetica 16 bold" )
    
    for x in range(165, 540, 75):
        canvas.create_line(375, x, 1050, x, fill = "#C0C0C0", width = 2)
    
    canvas.create_line(375, 90, 375, 540, fill = "#C0C0C0", width = 2)
        
    createScreen(canvas, data)
    
    translateCenterToScreen(data, canvas)
    
    if(data.fileMode):
        canvas.create_rectangle(150, 165, 1050, 395, fill = "white", width = 2)
        #17 lines
        if (len(data.lines) < 17):
            for x in range(len(data.lines)):
                canvas.create_text(155, 170+x*13, text = data.lines[x], anchor = "nw")
        else:
            for x in range(17):
                if( len(data.lines[data.textStart+x]) > 100 ):
                    temp_line = data.lines[data.textStart+x]
                    while(len(temp_line)):
                        index = min([100, len(temp_line)])
                        canvas.create_text(155, 170+x*13, text = temp_line[:index], anchor = "nw")
                        temp_line = temp_line[index:]
                        x+=1
                    x +=2
                else:
                    canvas.create_text(155, 170+x*13, text = data.lines[data.textStart+x], anchor = "nw")

#Convert an opencv image to a tkinter image, to display in canvas.                    
def opencvToTk(frame):
    
    pil_img = Image.fromarray(frame)
    tk_image = ImageTk.PhotoImage(image=pil_img)
    return tk_image

#manipulates frame
def frameManip(data):
    data.frame = cv2.resize(data.frame, (0,0), fx=0.30, fy=0.30)
    data.frame = cv2.flip(data.frame, 1) 
    data.gray = toGrayscale(data.frame)
    data.blur = toBlur(data.gray)
    data.thresh = toThreshold(data.blur)
    data.output = np.zeros(data.frame.shape, np.uint8)
    
    if data.firstIter:
        data.firstIter = False

#extracts features of contour    
def extractFeatures(data):
    data.listOfContours = extractContours(data.thresh)
    data.handContour = extractHandContour(data.listOfContours)
    data.handContourLen = extractContourLength(data.handContour)
    data.reducedContour = extractReducedContour(data.handContour, data.handContourLen)
    data.hull1, data.hull2 = extractConvexHull(data.handContour)
    data.defects = extractDefects(data.handContour, data.hull2)
    data.center = extractCentre(data.handContour, x0, y0)

#translates cursor to screen    
def translateCenterToScreen(data, canvas):
    (cX, cY) = data.center
    cX = (cX*data.ratio + 170)
    cY = (cY*data.ratio + 40)
    data.tkinterCenter = (cX, cY)
    canvas.create_oval(cX - 7, cY - 7, cX + 7, cY + 7, width = 2, fill = "#7c8864", outline = "white")

#displays features on frame    
def drawFeaturesOnFrame(data):
    drawContour(data.handContour, data.output)
    drawHull(data.hull1, data.output)
    if type(data.defects) != type(None):
        drawDefects(data.defects, data.handContour, data.output)
    drawCenter(data.center, data.output)
 
#draws camera
def drawCamera(canvas, data):
    data.tk_frame = opencvToTk(cv2.cvtColor(data.frame, cv2.COLOR_BGR2RGB))
    data.tk_output = opencvToTk(cv2.cvtColor(data.output, cv2.COLOR_BGR2RGB))
    data.tk_thresh = opencvToTk(data.thresh)
    canvas.create_image(data.width/6.2, data.height - data.height/6.6, image=data.tk_frame)
    canvas.create_image(data.width-data.width/6.2,data.height - data.height/6.6, image = data.tk_thresh)
    canvas.create_image(data.width/2, data.height - data.height/6.6, image = data.tk_output)

def gestureMove(data):
    #back
    if data.gStart[0] > 600 and data.gEnd[0] <400:
        if data.currPath == data.initPath:
            pass
        else: 
            path = ""
            list = data.currPath.split('/')
            for x in range (len(list)-1):
                path += "/" + list[x]
        
            data.currPath = path[1:]
            
            data.directory = os.listdir(data.currPath)
            data.directory = filterDirectories(data.directory)
            data.directory = createInitialVisibility(data)
            data.firstIteration = True
            data.fileMode = False
    #Up        
    elif(data.gStart[1] >350 and data.gEnd[1] <250):
        if data.fileMode:
            if (data.textStart!= 0):
                data.textStart -=1
        else:
            index = -1
            if data.directory[0][1] == 1:
                pass
            else:
                for x in range (len(data.directory)):
                    if data.directory[x][1] == 1:
                        index = x
                        break
            
                data.directory[index-1][1] = 1
                data.directory[index+2][1] = 0
                data.firstIteration = True
                
    #Down
    elif (data.gStart[1] <250 and data.gEnd[1] >350):
        if (data.fileMode):
            if (data.textStart + 17 != len(data.lines)):
                data.textStart +=1
        elif event.x >575 and event.x <625 and event.y > 410 and  event.y <435:
            index = -1
            if data.directory[-1][1] == 1:
                pass
            else:
                for x in range (len(data.directory)):
                    if data.directory[x][1] == 1:
                        index = x
                        break
                try:
                    data.directory[index][1] = 0
                    data.directory[index+3][1] = 1
                    data.firstIteration = True
                except: 
                    print("Whoops")
        

#for location mode    
def checkLocation(data):
    (cX, cY) = data.tkinterCenter
    index = -1
    for x in range (len(data.button)):
        dist = distance(cX, cY, data.button[x][1], data.button[x][2])
        if dist < data.radius:
            index = x
            break
            
    if index != -1:
        path = data.currPath + "/" +data.button[index][0]
        if os.path.isdir(path) == True:
            data.currPath += "/" +data.button[index][0]
        
            data.directory = os.listdir(data.currPath)
            data.directory = filterDirectories(data.directory)
            data.directory = createInitialVisibility(data)
            data.firstIteration = True
            data.locationMode = False
        else:
            try:
            
                file = open(path, "r")
                data.lines = file.readlines()
                data.fileMode = True
            except:
                pass
                
    #going back
    if cX >380 and cX <460 and cY > 112 and cY <143:
        if data.currPath == data.initPath:
            pass
        else: 
            path = ""
            list = data.currPath.split('/')
            for x in range (len(list)-1):
                path += "/" + list[x]
        
            data.currPath = path[1:]
            
            data.directory = os.listdir(data.currPath)
            data.directory = filterDirectories(data.directory)
            data.directory = createInitialVisibility(data)
            data.firstIteration = True
            data.fileMode = False
        
    #scroll
    #Up
    if data.fileMode and cX > 575 and cX <625 and cY >115 and cY <140:
        if (data.textStart!= 0):
            data.textStart -=1
    if cX>575 and cX <625 and cY >115 and cY <140:
        index = -1
        if data.directory[0][1] == 1:
            pass
        else:
            for x in range (len(data.directory)):
                if data.directory[x][1] == 1:
                    index = x
                    break
            
            data.directory[index-1][1] = 1
            data.directory[index+2][1] = 0
            data.firstIteration = True
    
    #Down 
    if data.fileMode and cX>575 and cX <625 and cY > 410 and  cY <435:
        if (data.textStart + 17 != len(data.lines)):
            data.textStart +=1
    elif cX >575 and cX <625 and cY > 410 and  cY <435:
        index = -1
        if data.directory[-1][1] == 1:
            pass
        else:
            for x in range (len(data.directory)):
                if data.directory[x][1] == 1:
                    index = x
                    break
            try:
                data.directory[index][1] = 0
                data.directory[index+3][1] = 1
                data.firstIteration = True
            except: 
                print("Whoops")

#called everytime a frame is recorded            
def cameraFired(data):
    """Called whenever new camera frames are available.
    Camera frame is available in data.frame. You could, for example, blur the
    image, and then store that back in data. Then, in drawCamera, draw the
    blurred frame (or choose not to).
    """
    frameManip(data)
    extractFeatures(data)
    drawFeaturesOnFrame(data)
    if(data.locationMode):
        checkLocation(data)
 
#called everytime the timer is called
def timerFired(data):
    if len(data.x0) > 10:
        data.x0 = data.x0[-10:]
        data.y0 = data.y0[-10:]
    
    if data.l:    
        if data.locationMode == False:
            data.time +=1
        
        if data.time%15 == 0:
            data.locationMode = True

#called when mouse is pressed  
def mousePressed(event, data):
    if data.fileMode:
        pass
    #opening a folder
    index = -1
    for x in range (len(data.button)):
        dist = distance(event.x, event.y, data.button[x][1], data.button[x][2])
        if dist < data.radius:
            index = x
            break
    if index != -1:
        path = data.currPath + "/" +data.button[index][0]
        if os.path.isdir(path) == True:
            data.currPath += "/" +data.button[index][0]
        
            data.directory = os.listdir(data.currPath)
            data.directory = filterDirectories(data.directory)
            data.directory = createInitialVisibility(data)
            data.firstIteration = True
        else:
            try:
                file = open(path, "r")
                data.lines = file.readlines()
                data.fileMode = True
            except:
                pass
            
    #going back
    if event.x >380 and event.x <460 and event.y > 112 and event.y <143:
        if data.currPath == data.initPath:
            pass
        else: 
            path = ""
            list = data.currPath.split('/')
            for x in range (len(list)-1):
                path += "/" + list[x]
        
            data.currPath = path[1:]
            
            data.directory = os.listdir(data.currPath)
            data.directory = filterDirectories(data.directory)
            data.directory = createInitialVisibility(data)
            data.firstIteration = True
            data.fileMode = False
        
    #scroll
    #Up
    if data.fileMode and event.x > 575 and event.x<625 and event.y >115 and event.y<140:
        if (data.textStart!= 0):
            data.textStart -=1
    if event.x > 575 and event.x<625 and event.y >115 and event.y<140:
        index = -1
        if data.directory[0][1] == 1:
            pass
        else:
            for x in range (len(data.directory)):
                if data.directory[x][1] == 1:
                    index = x
                    break
            
            data.directory[index-1][1] = 1
            data.directory[index+2][1] = 0
            data.firstIteration = True
    
    #Down 
    if data.fileMode and event.x >575 and event.x <625 and event.y > 410 and  event.y <435:
        if (data.textStart + 17 != len(data.lines)):
            data.textStart +=1
    elif event.x >575 and event.x <625 and event.y > 410 and  event.y <435:
        index = -1
        if data.directory[-1][1] == 1:
            pass
        else:
            for x in range (len(data.directory)):
                if data.directory[x][1] == 1:
                    index = x
                    break
            try:
                data.directory[index][1] = 0
                data.directory[index+3][1] = 1
                data.firstIteration = True
            except: 
                print("Whoops")

#called when a key is pressed
def keyPressed(event, data):
    if event.keysym == "q":
        data.root.destroy()
        
    if event.keysym == "s":
        data.startScreen = False
        
    if event.keysym == "l":
        data.l = not data.l
        data.locationMode = not data.locationMode
        
    if event.keysym == "g":
        
        if len(data.gStart) or len(data.gEnd):
            data.gEnd = data.tkinterCenter
            
        if not len(data.gStart):
            data.gStart = data.tkinterCenter
            
        print(data.gStart, data.gEnd)
        if len(data.gStart) and len(data.gEnd):
            gestureMove(data)
            data.gStart = []
            data.gEnd = []


#redraws everything
def redrawAll(canvas, data):
    if data.startScreen:
        canvas.create_rectangle(0,0, data.width, data.height, fill= "#686868", outline =  data.color[data.n%data.cLen], width =5)
   
        canvas.create_rectangle(30,30,data.width-30, 550, fill = data.color[data.n%data.cLen],outline = "", width = 0)
    
        drawCamera(canvas, data)
    
        canvas.create_line(0,0,0,data.height, width = 15, fill = data.color[data.n%data.cLen])
        canvas.create_line(0,0,data.width,0, width = 15, fill = data.color[data.n%data.cLen])
        canvas.create_line(0,data.height,data.width,data.height, width = 15, fill =     data.color[data.n%data.cLen])
        
    
        canvas.create_rectangle(150, 40, 1050, 540, fill = "white", width = 2, outline = "#C0C0C0")
        
        canvas.create_text(600, 320, text = "Press 'S' to start program", font ="Helvetica 24 bold")
        canvas.create_text(600, 220, text = "CVPy-EXPLORER", font ="Helvetica 104 bold")
        canvas.create_text(600, 394, text = "MODES:", font ="Helvetica 20 bold")
        canvas.create_text(600, 414, text = "MOUSE: Default",font ="Helvetica 20 bold", anchor = "n") 
        canvas.create_text(600, 434, text = "LOCATION: Press 'L'", font ="Helvetica 20 bold", anchor = "n")
        #canvas.create_text(600, 454, text = "GESTURE: Start location mode and start making gesture. Press 'g' mid-gesture.",font ="Helvetica 20 bold", anchor = "n")
    else:
        drawScreen(canvas, data)


#################################################
# Implements the program

def run(width=300, height=300):

    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.camera_index = 0

    data.timer_delay = 100 # ms
    data.redraw_delay = 50 # ms
    globalVariables(data, x0, y0)
    
    # Initialize the webcams
    camera = cv2.VideoCapture(data.camera_index)
    data.camera = camera

    # Make tkinter window and canvas
    data.root = Tk()
    canvas = Canvas(data.root, width=data.width, height=data.height)
    #data.img = ImageTk.PhotoImage(Image.open("Clippy.png"))
    canvas.pack()

    # Basic bindings. Note that only timer events will redraw.
    data.root.bind("<Button-1>", lambda event: mousePressed(event, data))
    data.root.bind("<Key>", lambda event: keyPressed(event, data))

    # Timer fired needs a wrapper. This is for periodic events.
    def timerFiredWrapper(data):
        # Ensuring that the code runs at roughly the right periodicity
        start = time.time()
        timerFired(data)
        end = time.time()
        diff_ms = (end - start) * 1000
        delay = int(max(data.timer_delay - diff_ms, 0))
        data.root.after(delay, lambda: timerFiredWrapper(data))

    # Wait a timer delay before beginning, to allow everything else to
    # initialize first.
    data.root.after(data.timer_delay, 
        lambda: timerFiredWrapper(data))

    def redrawAllWrapper(canvas, data):
        start = time.time()

        # Get the camera frame and get it processed.
        _, data.frame = data.camera.read()
        cameraFired(data)

        # Redrawing code
        canvas.delete(ALL)
        redrawAll(canvas, data)

        # Calculate delay accordingly
        end = time.time()
        diff_ms = (end - start) * 1000

        # Have at least a 5ms delay between redraw. Ideally higher is better.
        delay = int(max(data.redraw_delay - diff_ms, 5))
        data.root.after(delay, lambda: redrawAllWrapper(canvas, data))

    # Start drawing immediately
    data.root.after(0, lambda: redrawAllWrapper(canvas, data))

    # Loop tkinter
    data.root.mainloop()

    # Once the loop is done, release the camera.
    print("Releasing camera!")
    data.camera.release()

if __name__ == "__main__":
    run(1200,800)
    