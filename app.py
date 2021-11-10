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
#from boto3.s3.connection import S3Connection


# my two cat
X="Normal"
Y="Pneumonia"

ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg', 'gif'}
IMG_WIDTH, IMG_HEIGHT = 150,150
BUCKET_NAME='pneumoniadataset'

# Read the secret key from environment variables
aws_access_key_id = 'AKIAW5XAT2GRQHC3CH7U'
aws_secret_access_key = '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'

#create the website object
app = Flask(__name__)
#app.debug = True
app.config.from_pyfile('config.py')

def load_model_from_file():
    mySession = tf.compat.v1.Session()
    set_session(mySession)
    myModal = load_model('saved_model.h5')
    myGraph = tf.get_default_graph()
    return (mySession, myModal, myGraph)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#define the view for the top level page
@app.route('/', methods=['GET', 'POST'])

def upload_file():
    #Initial webpage load
    if request.method == 'GET' :
        return render_template('index.html')
    else: # if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            #print(file.filename)
            return redirect(request.url)
        
        file = request.files['file']
        #print(file.filename)
        # if user does not select file, browser may also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            #print(file.filename)
            return redirect(request.url)
        # If it doesn't look like an image file
        if not allowed_file(file.filename):
            #print(file.filename)
            flash('App only accept files of type'+str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        #When the user uploads a file with good parameters
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(filename)
            s3A = boto3.resource('s3',
                        aws_access_key_id=aws_access_key_id, #app.config.get('ACCESS_KEY', None),
                        aws_secret_access_key=aws_secret_access_key #app.config.get('SECRET_KEY', None)
                       # aws_session_token='secret token here'
                         )
            s3A.meta.client.upload_file(
                    Bucket = BUCKET_NAME,
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
            x = result[0][0] * 100 #10000
            per = "{:.2f}".format(x)
            #print(per)
            
            image_src = 'https://pneumoniadataset.s3.amazonaws.com/'+ filename
            if result[0][0] < 0.5 :
                answer = X + per + '%'
            else:
                answer = Y + per + '%'
            
            return render_template('index.html', result = answer, filename=filename, image_src=image_src)
        


def main():
    mySession, myModel,myGraph = load_model_from_file()
    
    #app.config['SECRET_KEY'] = 'development key'
    
    app.config['SESSION'] = mySession
    app.config['MODEL'] = myModel
    app.config['GRAPH'] = myGraph
    
    #app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 3*1024*1024 #3MB upload limit
    app.run()
                                                


#Launch everything
main()

 
