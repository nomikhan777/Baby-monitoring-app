import cv2
import mediapipe as mp
import math
import numpy as np
from yunet import YuNet
import time


class PoseEstimation():
    def __init__(self,
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = False,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.4):
        # False for Video feed
        self.mode = static_image_mode
        #
        self.comp = model_complexity
        # Smooth out the landmarks
        self.smooth = smooth_landmarks
        #
        self.enSeg = enable_segmentation
        #
        self.smoothSeg = smooth_segmentation
        # How sensitive the detection / tracking is (0.5)
        self.minDetect = min_detection_confidence
        self.minTrack = min_tracking_confidence

        # Initialising Pose Object with all variables / options needed for our use case
        # We can mess with model complexity and how sensitive the Framework is when we are closer to a finished prototype
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.comp, self.smooth, self.enSeg, self.smoothSeg, self.minDetect,
                                      self.minTrack)

    # Function needs more tweaking for drawing to be more accurate, not necessary for final program,
    # but useful for troubleshooting body tracking
    def drawLandmarks(self, image, draw=True):
        # Image is already being passed in RGB
        results = self.pose.process(image)
        # If Landmarks are detected
        if results.pose_landmarks:
            # if Draw flag is True (Draw Landmarks on Frame)
            if draw:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        # return trackingImage

    def babyPosition(self, image):
        baby_position = "Covered"
        image_height = 480
        image_width = 640
        output = image.copy()
        results = self.pose.process(output)
        # If body is not covered and can be tracked
        if results.pose_landmarks:
            lefteyeX = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].x
            righteyeX = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].x
            eyesPosition = righteyeX - lefteyeX
            # Tracking Z value of Baby's shoulders (left and right shoulders)
            leftShoulderZ = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z * image_height)
            rightShoulderZ = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z * image_width)
            # Comparing the difference in Z values and seeing how big difference between them is
            # If the 2 values are positive and negative (one positive and the other negative)
            # the baby is on its side as the difference between the Z values can't be that
            # large if the Baby is laying down flat (Face Up/ Face Down)
            # Therefore if the values are either both positive or both negative baby is not on its side
            if leftShoulderZ < 0 and rightShoulderZ > 0 or leftShoulderZ > 0 and rightShoulderZ < 0:
                differenceZ = -1 * (leftShoulderZ + rightShoulderZ)
                if differenceZ > 30 or differenceZ < -30:
                    baby_position = "On It's Side"
            elif leftShoulderZ > 0 and rightShoulderZ > 0 or leftShoulderZ < 0 and rightShoulderZ < 0:
                # as right eyes x value is greater than left eye x value
                # eyesposition will be positive
                # eyes are not visible
                # baby face down
                if eyesPosition > 0:
                    baby_position = "Face Down"
                # right eyes x value is lower than left eye x value
                # eyesposition will be negative
                # eyes are visible
                # baby face up
                elif eyesPosition < 0:
                    baby_position = "Face Up"
            return baby_position
        else:
            baby_position = "No Landmarks Detected"
            return baby_position


# I created a clas and put all the functions with out a class
# so it would be easer to get in main.py
class CameraOps:
    def __init__(self, camSource):
        self.camSource = camSource
        self.cap = None
        self.pose = PoseEstimation()

        self.toStream = True
        self.curFrame = None
        self.warning_message = ""
        self.warning_severity = 'LOW'  ## LOW/MEDIUM/HIGH

        self.load_stream()


    #====================================================================================
    def load_stream(self):
        self.cap = cv2.VideoCapture(self.camSource)
        ret, frame = self.cap.read()
        if ret:
            print("Cam/video load successful..")
            return True
        else:
            print("Unable to load camera / video stream, please check the source path..")
            return False



        #========================================================================================
    # Input is Face status and Body position
    def babyDangerWarning(self, face, body):
        # Going to use time to assign current status and then check it again a second or two later to make sure
        # it hasn't changed so no false warning won't be sent.
        # Have tried implementing this before with no success, if it still won't be a good way of doing it,
        # I will slow down the frame rate of the input to give tracking frameworks more time
        # to decide and hopefully be more accurate.
        oldtime = time.time()
        warning_severity = 'LOW'
        # If the face is Covered or Not detected and Body position is face down
        if face == "DANGER" and body == "Face Down":
            warning_severity = 'HIGH'
            return "WARNING baby in DANGER Face Down", warning_severity

        # If the face is Covered or Not detected and Body position can't be detected (Covered by a blanket)
        elif face == "DANGER" and body == "No Landmarks Detected":
            warning_severity = 'HIGH'
            return "WARNING baby in DANGER", warning_severity

        # If the Face is detected and Body position can be detected
        elif face == "Face detected" and body != "No Landmarks Detected":
            warning_severity = 'MEDIUM'
            return "Can detect face body can be detected", warning_severity

        # If the Face is detected but no Body
        elif face == "Face detected" and body == "No Landmarks Detected":
            warning_severity = 'LOW'
            return "Can detect face but no body", warning_severity

        # If the Face can be detected or Body positions are safe
        elif face == "Face Detected" or body == "Face Up" or body == "On It's Side":
            warning_severity = 'LOW'
            return "Everything is fine", warning_severity
        # if time.time() - oldtime > 1:
        elif face == "DANGER" and body == "Covered":
            warning_severity = 'HIGH'
            return "Cannot detect face body is Covered", warning_severity
        else:
            warning_severity = 'MEDIUM'
            return "Not sure", warning_severity


        #=============================================================================================
    def start_cam_stream(self):
        print("Started _Camera")
        self.toStream = True
        pose = PoseEstimation()
        # =================================================================================================
        wYuNet = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hYuNet = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # =================================================================================================
        # Location of Haar Cascade data sets in DataSets folder
        # Try using different one yourself, at first I thought "profileface" would
        # be useful to use with "frontalface" at the same time to detect when the doll is
        # on its side or face up, but "profileface" seem to barely function at all.
        # Don't know if that's due to the doll being too far away from the camera or just
        # how fast the doll changes positions, and it doesn't have enough time to detect the profile
        # =================================================================================================
        # Different Data Sets:
        # haarcascade_frontalface_default.xml
        # haarcascade_frontalface_alt.xml
        # haarcascade_profileface.xml
        #  cascadeFilePath = 'DataSets\haarcascade_frontalface_default.xml'
        # =================================================================================================
        while self.toStream:
            # print("Streaming...")
            success, image = self.cap.read()
            # True for night video, False for day video
            night = True
            if not success:
                print("No Frames to Track.")
                self.toStream = False
                break
            # =================================================================================================
            # Commented out drawing landmarks on the frame to improve performance
            # while testing different models for face detection
            # Also it's just a tool for visual aid of what's happening, won't be used in final program
            # bodyTrack = pose.drawLandmarks(image)
            pose.drawLandmarks(image)

            # =================================================================================================
            if night is True:
                src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                dst = cv2.equalizeHist(src)
                # cv2.imshow('Equalized Image', dst)
                rgb = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # faceTrackingHaarCascade(image, cascadeFilePath)
            # =================================================================================================
            # Tracking face
            face = self.faceTrackingYuNet(rgb, wYuNet, hYuNet)
            # Tracking baby position
            babyPosition = str(self.pose.babyPosition(rgb))
            # Warning message to be displayed on the frame
            print("babyPosition", babyPosition)
            print('face', face)
            self.warning_message, self.warning_severity = self.babyDangerWarning(face, babyPosition)
            print('self.warning_message', self.warning_message)
            print('self.warning_severity', self.warning_severity)
            # =================================================================================================
            # Flip the image horizontally for a selfie-view display.
            # image = cv2.flip(image, 1)
            image = cv2.putText(image, self.warning_message, (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2, cv2.LINE_AA)

            self.curFrame = image
            cv2.imshow('Position', image)
            # cv2.imshow('Body Track', bodyTrack)

            # Wait for 'q' key to stop the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def stop_cam_stream(self):
        self.toStream = False



    #==================================================================================================

    def faceTrackingHaarCascade(self, image, path):
        # Haar Cascade
        scale_factor = 1.2
        min_neighbors = 3
        min_size = (50, 50)
        webcam = False  # if working with video file then make it 'False'

        # Treat Detection
        status = False
        cascade = cv2.CascadeClassifier(path)

        # converting to gray image for faster video processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                         minSize=min_size)
        # if at least 1 face detected
        if len(rects) >= 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in rects:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #=====================================================================================================


    def visualize(self, image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
        # ====================================================================================================
        # Visualize Method from YuNet demo.py
        # https://github.com/opencv/opencv_zoo/blob/master/models/face_detection_yunet/demo.py
        output = image.copy()

        landmark_color = [
            (255, 0, 0),  # right eye
            (0, 0, 255),  # left eye
            (0, 255, 0),  # nose tip
            (255, 0, 255),  # right mouth corner
            (0, 255, 255)  # left mouth corner
        ]
        if fps is not None:
            cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

        for det in (results if results is not None else []):
            bbox = det[0:4].astype(np.int32)
            cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), box_color, 2)

            conf = det[-1]
            cv2.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                        text_color)

            landmarks = det[4:14].astype(np.int32).reshape((5, 2))
            for idx, landmark in enumerate(landmarks):
                cv2.circle(output, landmark, 2, landmark_color[idx], 2)

        return output

    #==============================================================================================================

    def faceTrackingYuNet(self, image, w, h):
        # Setting confidence Threshold of finding a face to 80% (confThreshold)
        # Setting number of faces to 1 instead of 5000 (topK)
        # Instantiating YuNet model
        model = YuNet(modelPath='DataSets/face_detection_yunet_2022mar.onnx',
                      inputSize=[320, 320],
                      confThreshold=0.80,
                      nmsThreshold=0.3,
                      topK=1,
                      backendId=3,
                      targetId=0)
        # Inference
        model.setInputSize([w, h])
        results = model.infer(image)  # results is a tuple

        for det in (results if results is not None else []):
            landmarks = det[4:14].astype(np.int32).reshape((5, 2))
            # If the face is detected face variable is set to faceDetected
            if det[-1]:
                face = "faceDetected"
            # Else the face variable is set to faceCovered
            else:
                face = "faceCovered"

        if results is None:
            face = "None"

        # Draw results on the input image
        frame = self.visualize(image, results)

        # Visualize results in a new Window
        cv2.imshow('Face Tracking yunet', frame)

        # Checking to see what state the face is and returning a status message
        # If face is covered or cannot be detected set and return message of DANGER
        if face == "faceCovered" or face == "None":
            message = "DANGER"
        # If face is detected set and return message of Face Detected
        elif face == "faceDetected":
            message = "Face Detected"
        # Else if previous checks fail set and return a message of program having difficulty
        else:
            message = "Having trouble detecting the face"
        return message
    # ====================================================================================================


