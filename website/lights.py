from flask import Blueprint, render_template, request, jsonify;
import sys
import os
sys.path.append(os.path.abspath("C:/Users/13039/Desktop/RINKY/RINKY_GRAD/sem 2/COMP 680_MAchine_learning/COMP680_Autonomous_Driving/traffic_signal"))
from detect_traffic_light import TrafficLights
from website.decode import decodeImage

auth = Blueprint('auth', __name__);

@auth.route('/traffic_light')
def traffic_lights():
    return render_template('trafficLights.html');


@auth.route('/predictLights', methods=["POST"])
def predict():
    print(request.method)
    print(request.form)
    if request.method == "POST":
        image = request.json['image']
        decodeImage(image, "inputImage.jpg")
        opimg = TrafficLights.traffic_lights("inputImage.jpg")
        return jsonify("Success")
