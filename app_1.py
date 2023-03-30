""" from flask import Flask, render_template, request, redirect, url_for
import os
import cnn_change 

from os.path import join, dirname, realpath
import csv
import pickle
import base64
import numpy as np
import cv2

app=Flask(__name__)

app.config["DEBUG"] = True

init_Base64 = 21;
conv = cnn_change.Conv3x3()
softmax = cnn_change.Softmax()
@app.route('/')
def index():
    return render_template('home_draw.html')


@app.route('/recognite', methods=['POST'])
def recognite():
        if request.method == 'POST':
               final_pred = "prediction message"
        global conv,softmax
        print("running")
        if cnn_change.check_model_files():
            print("model files exist")
            final_pred = None
            with open(os.getcwd()+'\conv', 'rb') as f_conv, open(os.getcwd()+'\softmax', 'rb') as f_soft:
                conv, softmax = pickle.load(f_conv), pickle.load(f_soft)
            for i in range(10):
                cnn_change.predict(cv2.imread(os.getcwd()+"\\"+str(i)+".png", cv2.IMREAD_GRAYSCALE))
        #Preprocess the image : set the image to 28x28 shape
        #Access the image
        draw = request.form['url']
        #Removing the useless part of the url.
        draw = draw[init_Base64:]
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        #Resizing and reshaping to keep the ratio.
        resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        # vect = np.asarray(resized, dtype="uint8")
        # vect = vect.reshape(1, 1, 28, 28).astype('float32')
        #Launch prediction
        # with open(os.getcwd()+'\conv', 'rb') as f_conv, open(os.getcwd()+'\softmax', 'rb') as f_soft:
        #     conv, softmax = pickle.load(f_conv), pickle.load(f_soft)
        #print(cnn_change.check_model_files())
        # cnn_change.load_model()
#        cnn_change.train_model()
#        cnn_change.save_model()
#        cv2.imshow("",resized)
#        cv2.waitKey(0)
        predictionResult = cnn_change.predict(resized)

        # my_prediction = model.predict(vect)
        #Getting the index of the maximum prediction
        # index = np.argmax(my_prediction[0])
        #Associating the index and its value within the dictionnary
        # final_pred = label_dict[index]

        return render_template('result.html', prediction =predictionResult)




if (__name__ == "__main__"):
   app.run(port = 5000) """