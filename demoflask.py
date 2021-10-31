# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:40:30 2021

@author: ayesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Prof. Donald Patterson (Westmont College)
    Twitter Contact: @djp3
"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello world, this is going to a web browser"

app.run()