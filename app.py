# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 23:17:34 2021

@author: Nidhi
"""
import os

from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import io
#import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from keras.backend import set_session
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
import cv2
import boto3

# my two cat
X="Normal"
Y="Pneumonia"

ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg', 'gif'}
IMG_WIDTH, IMG_HEIGHT = 150,150
BUCKET_NAME='pneumoniadatasetdemo'

#create the website object
app = Flask(__name__)
#app.debug = True
app.config.from_pyfile('app/config.py')
#print(app.config)
# Read the secret key from environment variables
# aws_access_key_id = app.config['ACCESS_KEY']
# aws_secret_access_key = app.config['SECRET_KEY']

# from boto.s3.connection import S3Connection
# s3 = S3Connection(os.environ['S3_KEY'], os.environ['S3_SECRET'])

def load_model_from_file():
    mySession = tf.compat.v1.Session()
    set_session(mySession)
    myModal = load_model('saved_model8.h5')
    #myGraph = tf.get_default_graph()
    myGraph = tf.compat.v1.get_default_graph()
    return (mySession, myModal, myGraph)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#define the view for the top level page
@app.route('/', methods=['GET', 'POST'])

def upload_file():
    #Initial webpage load
    if request.method == 'GET' :
        return render_template('index-poc.html')
    else: # if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        #print(file)
        # if user does not select file, browser may also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # If it doesn't look like an image file
        if not allowed_file(file.filename):
            flash('I only accept files of type'+str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        #When the user uploads a file with good parameters
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #print(filename)
            file.save(filename)
            
            image = cv2.imread(filename) #image.ndim
            #print(image)
            #print(image.ndim)
            ### splitting b,g,r channels
            b,g,r=cv2.split(image)

            ### getting differences between (b,g), (r,g), (b,r) channel pixels
            r_g=np.count_nonzero(abs(r-g))
            r_b=np.count_nonzero(abs(r-b))
            g_b=np.count_nonzero(abs(g-b))

            ### sum of differences
            diff_sum=float(r_g+r_b+g_b)

            ### finding ratio of diff_sum with respect to size of image
            ratio=diff_sum/image.size
            #print(ratio)
            if ratio>0.0:
                #print("image is color")
                flash('App only accept greyscale image')
                return redirect(request.url)
            
            else:
                s3A = boto3.resource('s3',
                            aws_access_key_id=app.config['ACCESS_KEY'],
                            aws_secret_access_key=app.config['SECRET_KEY']
                           # aws_session_token='secret token here'
                             )
                s3A.meta.client.upload_file(
                        Bucket = ayesha-pneumonia,
                        Filename=filename,
                        Key = filename
                    )
                #msg = "file is uploaded! "

                # get the image 
                object = s3A.Object(BUCKET_NAME, filename)

                img = object.get()['Body'].read()
                img = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))   
                img = img.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
                #img = np.dstack([img, img, img])  #stack 3 times
                img = img.astype('float32') / 255
                #img = np.expand_dims(img, axis=0)
                mySession,myModel,myGraph= load_model_from_file()

                result = myModel.predict(img)
                #print(result)
                res_normal = 1 - result[0][0]
                #print(res_normal)
                x = result[0][0] * 100 #10000
                x_normal = res_normal * 100
                per = "{:.2f}".format(x)
                per_normal = "{:.2f}".format(x_normal)
                #print(per)

                # image_src = 'https://.s3.amazonaws.com/'+ filename
                image_src = 'https://pneumoniadatasetdemo.s3.amazonaws.com/'+ filename

                if result[0][0] < 0.5 :
                    answer = 'This x-ray image has a ' + per_normal + '% chance of being normal/non-pneumonia.'
                else:
                    answer = 'This image has a ' + per  + '% chance of being positive for pneumonia.'

                return render_template('index-poc.html', result = answer, filename=filename, image_src=image_src)
      


def main():
    mySession, myModel,myGraph = load_model_from_file()
    
    #app.config['SECRET_KEY'] = 'development key'
    
    app.config['SESSION'] = mySession
    app.config['MODEL'] = myModel
    app.config['GRAPH'] = myGraph
    
    #app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024 #16MB upload limit
    if __name__ == "__main__":
        app.run(debug=True)
    


#Launch everything
main()


