import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('v004.mp4')
pre_time = 0
while True:
    succes, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    # print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w) , int(lm.y*h)
                print(id, cx, cy)

                if id==4:
                    cv2.circle(frame, (cx, cy), 10, (159, 19, 75), cv2.FILLED)

    current_time = time.time()
    fps = 1/ (current_time-pre_time)
    pre_time = current_time

    # # Black Box
    # start_point = (0, 0)
    # end_point = (102, 32)
    # color = (255, 255, 255)
    # thickness = -1

    # black_box = cv2.rectangle(frame, start_point, end_point, color, thickness) 

    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

    cv2.putText(
        frame,
        str(int(fps)),
        (70, 50),
        font, 1,
        (255, 200, 0),
        2
    )

    cv2.imshow("Camera", frame)
    if cv2.waitKey(10) == ord('0'):
        break

    cap.release()
    cv2.destroyAllWindows()