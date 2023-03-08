import cv2
import mediapipe as mp
import time

vdo = 'v00.mp4'
cap = cv2.VideoCapture(0)
pre_time = 0
current_time = 0

mpFaceDetect = mp.solutions.face_detection
faceDetection = mpFaceDetect.FaceDetection()
mpDraw = mp.solutions.drawing_utils

while True:
    succes, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(frameRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
                #### It is default Face Detection
                # mpDraw.draw_detection(frame, detection)

                # print(id, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)

                #### My creation
                bboxC = detection.location_data.relative_bounding_box
                frame_h, frame_w, frame_c = frame.shape
                bbox = int(bboxC.xmin * frame_w), int(bboxC.ymin * frame_h), \
                    int(bboxC.width * frame_w), int(bboxC.height * frame_h)
                
                cv2.rectangle(frame, bbox, (255, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{int(detection.score[0]*100)}%",
                    (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 255),
                    2
                )

    current_time = time.time()
    fps = 1/ (current_time-pre_time)
    pre_time = current_time

    # FPS Box
    start_point = (0, 0)
    end_point = (105, 32)
    color = (50, 0, 0)
    thickness = -1
    FPS_Box = cv2.rectangle(frame, start_point, end_point, color, thickness) 
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

    # FPS Mature
    cv2.putText(
        FPS_Box,
        str(int(fps)),
        (64, 24),
        font, 0.8,
        (255, 200, 0),
        2
    )

    # FPS Text
    cv2.putText(
        FPS_Box,
        'FPS:',
        (8, 22),
        cv2.FONT_HERSHEY_TRIPLEX, 0.7,
        (255, 200, 0),
        1
    )

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('0'):
        break
    
cap.release()
cv2.destroyAllWindows() 