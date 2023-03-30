import cloudpickle as pickle
import os
import cv2
import cnn_change
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
immm = cv2.imread("0.png",cv2.IMREAD_GRAYSCALE)
conv = cnn_change.Conv3x3()
softmax = cnn_change.Softmax()
with open(os.getcwd()+'\conv', 'rb') as f_conv, open(os.getcwd()+'\softmax', 'rb') as f_soft:
    conv, softmax = pickle.load(f_conv), pickle.load(f_soft)                
for i in range(10):
    cnn_change.predict(cv2.imread(os.getcwd()+"\\"+str(i)+".png", cv2.IMREAD_GRAYSCALE),conv,softmax)
cnn_change.predict(immm,conv,softmax)