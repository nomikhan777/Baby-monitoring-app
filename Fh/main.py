from threading import Thread
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import time
import json
from traking import PoseEstimation, CameraOps

################### INPUTS HERE ################################
camera_source = 'baby_aboveNight2.avi'  # put "0" for webcam

################### INPUTS HERE ################################


camera = CameraOps(camSource=camera_source)

app = Flask(__name__)
message_sent = False

##########   GLOBAL variables ##########
detected_data = {"warning_message": "", "warning_severity": "LOW"}


################################################### DETECTION [START]########################################################
@app.route("/getResult")
def getResult():
    """Function called from javascript after every 0.5 seconds
    in id_assigning.html for displaying person recognized and QR code"""

    object_json = json.dumps(detected_data)

    return object_json


def get_detection_result():
    """This function yeilds frames for streaming on detection/ recognition page. Also it stores realtime
    face detected person name and QR code values in dictionary called detected_data"""
    global detected_data
    timest = time.time()

    t1 = Thread(target=camera.start_cam_stream)
    t1.start()
    if camera.toStream == True:
        while True:
            if camera.curFrame is not None:
                ret, buffer = cv2.imencode('.jpg', camera.curFrame)
                detected_data["warning_message"] = camera.warning_message
                detected_data["warning_severity"] = camera.warning_severity

                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            else:
                time.sleep(0.5)
            if camera.toStream == False:
                break


@app.route("/vid_stream")
def vid_stream():
    """ Function called from id_assigning.html to render steam to show on web page """
    return Response(get_detection_result(), mimetype='multipart/x-mixed-replace; boundary=frame')


################################################### DETECTION [END] ########################################################
@app.route("/stop_all_cameras", methods=['POST'])
def stop_all_cameras():
    global camera
    if camera is not None:
        camera.toStream = False

    return render_template('index.html')


@app.route('/start_stream', methods=['post'])
def start_stream():
    global camera
    if camera is not None:
        camera.toStream = True

    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    global camera
    stop_all_cameras()
    camera.toStream = True

    return render_template('index.html')


if __name__ == "__main__":
    # app.run(debug=True, host="192.168.1.225")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)

