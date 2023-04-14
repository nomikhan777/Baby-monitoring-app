import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture('babyPositions.avi')
#file = 'BabyFramesV2\\baby_face.jpg'
#cap = cv2.imread(file)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    image_height = 480
    image_width = 640
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    #My own bit of code to get the Coordinates of Nose And Left Hip
    if not results.pose_landmarks:
        continue
    noseX = '%.2f' % (results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width)
    noseY = '%.2f' % (results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height)
    #Left Hip X and Y Coordinates
    L_hipX = '%.2f' % (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width)
    L_hipY = '%.2f' % (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height)
    #Left Knee X and Y Coordinates
    TestingLandMarkX = '%.2f' % (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width)
    TestingLandMarkY = '%.2f' % (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height)

    noseZ = '%.2f' % (results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z * image_width)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (0, 30)
    # fontScale
    fontScale = 0.8
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2

    coordinates = f'Nose:{str(noseX)}X {str(noseY)}Y ' \
                  f'Nose Z: {str(noseZ)}Z '
    # Using cv2.putText() method
    #image = cv2.putText(image, Coordinates, org, font,
    #                    fontScale, color, thickness, cv2.LINE_AA)
    print(
        f'({coordinates}) '
        f'Left Knee:{str(TestingLandMarkX)}X {str(TestingLandMarkY)}Y '
        f' Z:{str(noseZ)}Z'

        #f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width},'
        #f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height},'
        #f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z * image_width})'

    )
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)
    # Using cv2.putText() method to display Nose Coordinates X and Y
    image = cv2.putText(image, coordinates, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('MediaPipe Pose', image)

    # Wait for 'q' key to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
