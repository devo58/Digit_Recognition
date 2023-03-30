from flask import Flask, render_template, request, redirect, url_for
import os
import cnn_change 

from os.path import join, dirname, realpath
import csv
import pickle
import cloudpickle as pickle
import base64
import numpy as np
import cv2


conv = cnn_change.Conv3x3()
softmax = cnn_change.Softmax()


app=Flask(__name__)

app.config["DEBUG"] = True

init_Base64 = 21;

@app.route('/')
def index():
    return render_template('home_draw.html')


@app.route('/recognite', methods=['POST'])
def recognite():
        if request.method == 'POST':
               final_pred = "prediction message"
        final_pred = None
        print("running")
        with open(os.getcwd()+'\conv', 'rb') as f_conv, open(os.getcwd()+'\softmax', 'rb') as f_soft:
            conv, softmax = pickle.load(f_conv), pickle.load(f_soft)                
#        for i in range(10):
#            cnn_change.predict(cv2.imread(os.getcwd()+"\\"+str(i)+".png", cv2.IMREAD_GRAYSCALE),conv,softmax)
        #Preprocess the image : set the image to 28x28 shape
        #Access the image
        draw = request.form['url']
        #Removing the useless part of the url.
        draw = draw[init_Base64:]
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        #image[np.where((image != [0,0,0]).all(axis = 2))] = [255,255,255]
        #Resizing and reshaping to keep the ratio.
        image[image != 0] = 255
        resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        print(resized)
        print(np.any(resized!=[0,0,0],axis=-1))
        # vect = np.asarray(resized, dtype="uint8")
        # vect = vect.reshape(1, 1, 28, 28).astype('float32')
        #Launch prediction
        # with open(os.getcwd()+'\conv', 'rb') as f_conv, open(os.getcwd()+'\softmax', 'rb') as f_soft:
        #     conv, softmax = pickle.load(f_conv), pickle.load(f_soft)
#        kernel = np.ones((3,3), np.uint8)
#        resized = cv2.dilate(resized,kernel,iterations = 1)
        print(cnn_change.check_model_files())
        # cnn_change.load_model()
#        cnn_change.train_model()
#        cnn_change.save_model()
#        cv2.imshow("abc",resized)
        cv2.imwrite("abcsadsadas.png",resized)
#        cv2.waitKey(0)
        predictionResult = cnn_change.predict(resized,conv,softmax)

        # my_prediction = model.predict(vect)
        #Getting the index of the maximum prediction
        # index = np.argmax(my_prediction[0])
        #Associating the index and its value within the dictionnary
        # final_pred = label_dict[index]

        return render_template('result.html', prediction =predictionResult)




if (__name__ == "__main__"):
     app.run(port = 5000)