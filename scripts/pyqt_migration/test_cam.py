#!/usr/bin/env python

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import cv2
import pyqtgraph.ptime as ptime

app = QtGui.QApplication([])
cam = cv2.VideoCapture(0)

## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
view = win.addViewBox()

## lock the aspect ratio so pixels are always square
view.setAspectLocked(True)

## Create image item
img = pg.ImageItem(border='w')
view.addItem(img)

## Set initial view bounds
view.setRange(QtCore.QRectF(0, 0, 640, 480))

updateTime = ptime.time()
fps = 0

def Update():
    global img, updateTime, fps, cam
    
    #take a frame
    ret, frame = cam.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img.setImage(cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE))
    QtCore.QTimer.singleShot(1, Update)
    #now = ptime.time()
    #fps2 = 1.0 / (now-updateTime)
    #updateTime = now
    #fps = fps * 0.9 + fps2 * 0.1
    #print(fps)

Update()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()













