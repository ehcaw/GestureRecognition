from camera import VideoCamera
from flask import Flask, render_template, Response

app = Flask(__name__)

def gen_frames(camera):  
    """
    function to get frames
    """
    while True:
        #get camera frame
        jpeg = camera.convert_jpeg()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')

def gen_labels(camera):
    """
    function to get labels
    """
    while True:
        label = camera.get_label()
        # make sure were actually processing something
        print('label generated')
        print(label)
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
