from camera import VideoCamera
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)


def gen_frames(camera):  
    """
    function to get frames
    """
    while True:
        #get camera frame
        frame = camera.get_frame()
        # print(frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/about')
def hello():
    return render_template('./about.html')


if __name__ == "__main__":
    app.run(debug=True)
