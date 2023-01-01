import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import os

sio = socketio.Server()

app = Flask(__name__) #'__main__'
speed_limit = 16

# (see explaination in tut9 at [140])
def img_preprocess(img):
    img = img[60:135,:,:] 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


@sio.on('telemetry') # remote control event for the car on the simulator
def telemetry(sid, data):
    current_speed = float(data['speed'])

    # get current image from the simulator that sends to our server
    # byteIO is used to mimic data like a normal file for opening
    # use that before Image.open
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image) # change to numpy array
    image = img_preprocess(image)
    inputs = np.array([image]) # convert to inputs to put in the model

    steering_angle = float(model.predict(inputs))
    throttle = 1.0 - current_speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, current_speed))
    
    # send back to the simulator
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0) # initial value

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

@sio.on('disconnect')
def disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    currentDir = os.path.dirname(os.path.abspath(__file__))
    modelFilePath = os.path.join(currentDir,'../behavioral_cloning/self_driving_car_model_3.h5')

    model = load_model(modelFilePath)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


