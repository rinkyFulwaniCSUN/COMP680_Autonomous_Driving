from website import create_app;

app = create_app();
if __name__ == '__main__':
    app.run(debug=True);

app.config['IMAGE_UPLOAD'] = 'C:/Users/13039/Desktop/RINKY/RINKY_GRAD/sem 2/COMP 680_MAchine_learning/COMP680_Autonomous_Driving/traffic_sign/test_images'
