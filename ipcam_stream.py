import cv2
import os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

#cap = cv2.VideoCapture('http://192.168.1.239:81/videostream.cgi?user=admin&pwd=')
#Switched internet provider and had to reconfigured the camera with a new ip address for it to work using TENVCIS IP Camera Search Tool
cap = cv2.VideoCapture('http://192.168.178.42:81/videostream.cgi?user=admin&pwd=')

# print("After URL")

while True:

    # print('About to start the Read command')
    ret, frame = cap.read()
    edge = cv2.Canny(frame, 100, 50)
    # print('About to show frame of Video.')
    cv2.imshow("Capturing", frame)
    # print('Running..')

    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Name of the stored image
        name = 'baby_side_pacifier.jpg'
        os.chdir('BabyFramesV2')

        # Storing the image using the imwrite function
        cv2.imwrite(name, frame)
        break


    #elif cv2.waitKey(1) & 0xFF == ord('q'):
    #   break

cap.release()
cv2.destroyAllWindows()


