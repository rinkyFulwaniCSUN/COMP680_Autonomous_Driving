import cv2
import numpy as np
import object_detection
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import os

class TrafficLights:


    def traffic_lights(filename):
        FILENAME = "test_traffic_light/1.jpg"
    # Load the Inception V3 model
        model_inception = InceptionV3(weights='imagenet', include_top=True, input_shape=(299,299,3))

    # Resize the image
        img = cv2.resize(preprocess_input(cv2.imread(filename)), (299, 299))

    # Generate predictions
        out_inception = model_inception.predict(np.array([img]))

    # Decode the predictions
        out_inception = imagenet_utils.decode_predictions(out_inception)

        print("Prediction for ", filename , ": ", out_inception[0][0][1], out_inception[0][0][2], "%")

    # Show model summary data
        model_inception.summary()

    # Detect traffic light color in a batch of image files
        files = object_detection.get_files('test_traffic_light/*.jpg')

    # Load the SSD neural network that is trained on the COCO data set
        model_ssd = object_detection.load_ssd_coco()

    # Load the trained neural network
        model_traffic_lights_nn = keras.models.load_model(os.getcwd() + "/traffic_signal/trafficSinal.h5")
        opimg, output, opfilename =  object_detection.perform_object_detection(
        model_ssd, filename, save_annotated=True, model_traffic_lights=model_traffic_lights_nn)
        print(output)
        return opimg;
