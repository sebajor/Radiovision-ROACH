#!/usr/bin/env python

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import cv2
from matplotlib.cm import get_cmap
import time
from artificial_data import radiovision2
import ipdb
import sys
import traceback

class Workersignal(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(tuple)
    #prorgress = QtCore.pyqtSignal(tuple)


class Worker(QtCore.QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = Workersignal()
        #self.kwargs['progress_callback'] = [0,0]

    @QtCore.pyqtSlot()   
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signal.error.emit((0, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)


class Sim(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(Sim, self).__init__()
        self.cp = get_cmap('inferno')
        self.cam = cv2.VideoCapture(0)
        self.win = pg.GraphicsLayoutWidget()
        self.win.show()
        self.view = self.win.addViewBox()
        ret, frame = self.cam.read()
        self.view.setAspectLocked(True)

        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        self.view.setRange(QtCore.QRectF(0, 0, 640, 480))
            
        self.threadpool = QtCore.QThreadPool()
        self.threadpool.setMaxThreadCount(2) #n of threads running
        ##modify when add the marker feature
 
        self.loc = [2,2]

        self.cam_frame = np.zeros([480,640,3]).astype(np.uint8)
        self.roach_frame = np.zeros([480,640,3]).astype(np.uint8)
        self.superposition = np.zeros([480,640,3]).astype(np.uint8)
        self.Update()
    
    def Update(self):
        ###Solo un thread puede usar la camara a la vez, sino tira
        ###error.. hay que manejar eso

        #ipdb.set_trace()
        
        self.superposition = cv2.addWeighted(self.cam_frame,0.7,self.roach_frame,0.4,0)
        self.img.setImage(cv2.rotate(self.superposition,cv2.ROTATE_90_CLOCKWISE)) 
        QtGui.QApplication.processEvents()
        roach_up = Worker(self.roach_data)
        cam_up = Worker(self.cam_data)

        roach_up.signals.result.connect(self.plot_roach)
        cam_up.signals.result.connect(self.plot_cam)
        
        self.threadpool.tryStart(cam_up)
        self.threadpool.start(roach_up) 
        QtCore.QTimer.singleShot(1, self.Update)
 

    def roach_data(self, *args):
        self.loc[0] = self.loc[0]+(np.random.random()-0.5)*0.1
        self.loc[1] = self.loc[1]+(np.random.random()-0.5)*0.1
        if (self.loc[0]>3 or self.loc[0]<0):
            self.loc[0]=0
        if (self.loc[1]<0 or self.loc[1]>3):
            self.loc[1] = 3
        data_sim = radiovision2([int(self.loc[0]), int(self.loc[1])])
        data = cv2.resize(data_sim, dsize=(640,480), interpolation=cv2.INTER_CUBIC)
        data = self.cp(data)
        data = (data[:,:,0:3]*255).astype(np.uint8)
        return (data,1)
        
    def cam_data(self, *args):
        #ipdb.set_trace()
        ret, frame = self.cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        return (frame,2)

    def plot_roach(self, *args):
        #ipdb.set_trace()
        self.roach_frame = args[0][0]
    
    def plot_cam(self, *args):
        self.cam_frame = args[0][0]

    
    
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = QtGui.QApplication([])
        window = Sim()
        app.exec_()
