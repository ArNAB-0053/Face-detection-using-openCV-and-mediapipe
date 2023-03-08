import cv2
# import mediapipe as mp
import time
import Face_Detection as Fd

cap = cv2.VideoCapture(0)
pre_time = 0
current_time = 0
detector = Fd.FaceDetector()
while True:
    succes, frame = cap.read()
    frame, bboxs = detector.findFace(frame)
    current_time = time.time()
    fps = 1 / (current_time-pre_time)
    pre_time = current_time

    # FPS Box
    start_point = (0, 0)
    end_point = (110, 32)
    color = (50, 0, 0)
    thickness = -1
    FPS_Box = cv2.rectangle(
        frame, start_point, end_point, color, thickness)
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
