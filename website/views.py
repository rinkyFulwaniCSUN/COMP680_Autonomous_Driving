from flask import Blueprint, render_template, request;
import sys
import os
sys.path.append(os.path.abspath("C:/Users/13039/Desktop/RINKY/RINKY_GRAD/sem 2/COMP 680_MAchine_learning/COMP680_Autonomous_Driving/traffic_sign"))
from execute_traffic_sign import trafficsign
#import trafficSign;


views = Blueprint('views', __name__)

@views.route('/')
def home():
    return render_template('base.html');

@views.route('/traffic_sign')
def traffic_sign():
    return render_template('traffic_sign.html', result=[{}]);

@views.route('/train_traffic_sign', methods=["GET", "POST"])
def train_traffic_sign():
    if request.method == "POST":
        pass
        #trafficSign(request.form.epochs);
        #return render_template('success.html')
    return render_template('train_traffic_sign.html');

@views.route('/predict', methods=["POST"])
def predict():
    print(request.method)
    print(request.form)
    if request.method == "POST":
        result = trafficsign()
        print("from views.py==", result)
        return render_template('traffic_sign.html', result=result)
    else:
        return render_template('traffic_sign.html', result='')
