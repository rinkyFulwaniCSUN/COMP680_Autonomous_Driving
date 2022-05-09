from flask import Blueprint, render_template, request;
import sys
import os
sys.path.append(os.path.abspath("C:/Users/13039/Desktop/RINKY/RINKY_GRAD/sem 2/COMP 680_MAchine_learning/COMP680_Autonomous_Driving/traffic_signal"))
import detect_traffic_light

auth = Blueprint('auth', __name__);

@auth.route('/traffic_light')
def traffic_lights():
    return render_template('traffic_lights.html', result=[{}]);


@auth.route('/predictLights', methods=["POST"])
def predict():
    print(request.method)
    print(request.form)
    if request.method == "POST":
        result = detect_traffic_light.traffic_lights()
        print("from views.py==", result)
        return render_template('traffic_lights.html', result=result)
    else:
        return render_template('traffic_lights.html', result='')
