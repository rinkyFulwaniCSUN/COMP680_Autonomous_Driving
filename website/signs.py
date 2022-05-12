from flask import Blueprint, render_template, request, jsonify;
from flask_cors import CORS, cross_origin
import sys
import os
sys.path.append(os.path.abspath("C:/Users/13039/Desktop/RINKY/RINKY_GRAD/sem 2/COMP 680_MAchine_learning/COMP680_Autonomous_Driving"))
from traffic_sign.execute_traffic_sign import trafficSign
from website.decode import decodeImage
#from werkzeug.utils import secure_filename

#import trafficSign;
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

views = Blueprint('views', __name__)

@views.route('/')
def home():
    return render_template('base.html');

@views.route('/traffic_sign')
def traffic_sign():
    return render_template('index.html', result=[{}]);



@views.route('/predict', methods=["POST"])
def predict():
    print(request.method)
    print(request.form)
    print(request.files)

    if request.method == "POST":

        # image = request.files['file']
        # print(image)
        # #filename = secure_filename(image.filename)
        # basedirectory = os.path.abspath(os.path.dirname(__file__))
        # image.save(os.path.join(basedirectory, app.config['IMAGE_UPLOAD'], image.filename))
        image = request.json['image']
        decodeImage(image, "inputImage.jpg")
        result = trafficSign.trafficsign("inputImage.jpg")
        print("from views.py==", result)
        return jsonify(result)
        #return render_template('traffic_sign.html', result=jsonify(result))
    else:
        return render_template('traffic_sign.html', result='')


@views.route('/train_traffic_sign', methods=["GET", "POST"])
def train_traffic_sign():
    if request.method == "POST":
        pass
        #trafficSign(request.form.epochs);
        #return render_template('success.html')
    return render_template('train_traffic_sign.html');
