from camera import VideoCamera
from flask import Flask, render_template, Response
import cv2
import time

app = Flask(__name__)

camera = cv2.VideoCapture(0)


def gen_frames(camera):  
    """
    function to get frames
    """
    i = 0
    while True:
        #get camera frame
        frame, label = camera.get_frame()
        if i % 100 == 0: 
            print(label)
        i += 1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        #return label
def gen_labels(camera):
    while True:
        label = camera.get_label()
        print('label generated')
        return label

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_label')
def get_label():
    return Response(gen_labels(VideoCamera()), mimetype='text/plain')

@app.route('/about')
def hello():
    return render_template('./about.html')


if __name__ == "__main__":
    app.run(debug=True)
