from flask import Blueprint, render_template, request, jsonify;
from flask_cors import CORS, cross_origin
import sys
import os
sys.path.append(os.path.abspath("C:/Users/13039/Desktop/RINKY/RINKY_GRAD/sem 2/COMP 680_MAchine_learning/COMP680_Autonomous_Driving"))
from website.decode import decodeImage
#from werkzeug.utils import secure_filename

#import trafficSign;
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

object = Blueprint('object', __name__)

@object.route('/object_output')
def home():
    return render_template('object.html');
