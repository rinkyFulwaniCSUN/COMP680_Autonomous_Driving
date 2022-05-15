The technology of autonomous driving has been attracting attention of both academia and industry over the last several years. Generally, the studies on autonomous driving could be divided into three different components including perception, decision-making, and control. The perception component aims to efficiently obtain environmental information for autonomous vehicles, and the control component aims to set appropriate mechanical parameters so that the autonomous vehicles could travel forward by following the planned path.The decision-making component tries to determine a perfect path for autonomous vehicles, which could safely navigate the vehicles toward their destinations. Therefore, it is the core component of the three components because it determines all the actions of the autonomous vehicle, such as lane change, brake, and acceleration.
The aim is to develop an algorithm that can be used for Autonomous driving in cars. The important aspects of this project are to visualize all aspects of the road while driving a car.  Lane detection, traffic signs, traffic signals, pedestrians and other obstacles like cars, objects, and so on, are the important aspects to visualize a road properly. These aspects will help keep the driver and other passengers safe.
Our goals for this project include lane detection, pedestrian detection, obstacle detection, and traffic signs and traffic lights classification. The autonomous vehicle must reliably detect the boundaries of its current lane for accurate localization to guarantee the driving safety. Moreover, diverse lane patterns, such as solid, broken, splitting, and merging lanes, make independent lane modeling difficult. Traditionally, algorithms based on highly specialized handcrafted features were used to solve these problems. In these algorithms, the handcrafted features include color-based features and the structure tensor. There are some assumptions on lanes in existing lane detection methods, such as lanes are parallel, and lanes are straight or close to straight. However, these assumptions are not always valid, especially in urban situations. Recently, methods based on deep neural networks were applied to detect lanes for autonomous vehicles. The convolutional neural networks (CNNs) had achieved promising results. As a result, we have mostly used CNN to train the model for various detections. The images are extracted, detected, and recognized by using image processing techniques, such as threshold techniques, and so on.
We are planning to use OpenCV (Computer Vision) for the image processing part. We will be using a dataset from the open source platform and we will be deploying our machine learning algorithm on that dataset to determine different aspects of the roads. Finally we will use CNN to predict those aspects.


##
Setup:

The following is the setup that we need to do on our system in order to set up the GPU for tensorflow.

##
We have used Miniconda to setup our environment for this project:
  Download Link: https://docs.conda.io/en/latest/miniconda.html

##
You should install the latest version of your GPUs driver. You can download drivers here:
  https://www.nvidia.com/Download/index.aspx

##
You will need Visual Studio, with C++ installed. By default, C++ is not installed with Visual Studio, so make sure you select all of the C++ options:
  https://www.tensorflow.org/install/gpu

##
Then download that (or a later) version of CUDA from the following site:
  https://developer.nvidia.com/cuda-downloads

##
CuDNN:
  https://developer.nvidia.com/cudnn

##
Jupyter(Use the following command on anaconda prompt):
  conda install -y jupyter

##
Setup the environment on Anaconda:
  conda create -y --name tensorflow python=3.9

##
To enter this environment, you must use the following command (for Windows), this command must be done every time you open a new Anaconda/Miniconda terminal window:
  conda activate tensorflow

##
Jupyter Kernel:
It is easy to install Jupyter notebooks with the following command:
  conda install -y jupyter

##
Once Jupyter is installed, it is started with the following command:
  jupyter notebook

##
Step 9: Install TensorFlow/Keras
  pip install tensorflow

##
Testing if GPU is available:
  import tensorflow as tf
  print(tf.__version__)
  print(len(tf.config.list_physical_devices('GPU'))>0)




##
Installation:

1. Install python3 to your local computer.
2. Clone the github repository to a local folder. This folder will be named as "COMP680_Autonomous_Driving" 
3. Open the terminal and go the location where you cloned the github repository.
4. Once you have installed miniconda you can open the miniconda prompt and set up the environment.
5. There is a requirements.yml file which you need to execute in order to set up the environment with all the required dependencies.
6. The following are the commands that needs to be executed on the miniconda terminal:
    conda env create --file requirements.yml
    conda activate py37
    Now the tensorflow envuronment is set up and it is activated. All dependencies and changes will be restricted to this environment and will not reflect on other 
    existing environments.
7. Once all the dependencies are installed, you have to start the server to start the web application. For this step, run the command "python main.py"
8. If this command is executed successfully, it will show you the link for web app. Open this link in your preferred browser and check out the application. 
9. For Traffic Sign module, once you upload the image and click on Predict button, it will show the output label of the image on the webpage itself. Sample images to upload and test are provided under "traffic_sign/test_images" folder.
10. To execute traffic signal detection, after you upload the image and click on Predict button, it will save the output result image "inputImage_test.jpg" in the root folder. The prediction can be seen there. 
11. To execute the lane detection module you just have to click on the lane detection button on the app and it will be executed which will display the lane being detected in real time.
12. To execute the vehicle and pedestrian detection module you just have to click on the Vehicle and Pedestrian Detection button on the app and it will be executed which will show the vehicles and pedestrians being detected in real time.


