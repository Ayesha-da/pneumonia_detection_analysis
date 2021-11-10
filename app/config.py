# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 10:48:13 2021

@author: Nidhi
"""
from os import environ, path
#from r-dotenv import load_dotenv


#basedir = path.abspath(path.dirname(__file__))
#(path.join(basedir, '.env'))

#FLASK_ENV = 'development'
class Config:
    ACCESS_KEY = environ.get('ACCESS_KEY')
    SECRET_KEY = environ.get('SECRET_KEY')
