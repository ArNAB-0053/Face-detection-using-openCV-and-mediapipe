import cv2
import mediapipe as mp
import time

class PoseEstimation():

    def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.static_image_mode,
            self.model_complexity,
            self.smooth_landmarks,
            self.enable_segmentation,
            self.smooth_segmentation,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
         
    def findPose(self, frame, draw = True):     
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(frameRGB)
        # print(results.pose_landmarks)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return frame        

    def position(self, frame, draw = True):
            lmList = []

            if self.results.pose_landmarks:
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x*w) , int(lm.y*h)
                        lmList.append([id, cx, cy])

                        if draw:
                            cv2.circle(frame, (cx, cy), 2, (159, 19, 75), cv2.FILLED)

            return lmList 

def main():
    cap = cv2.VideoCapture('v004.mp4') # just give 0 in the braket then it will show as a live camera
    pre_time = 0

    detector = PoseEstimation()
    while True:
        succes, frame = cap.read()
        frame = detector.findPose(frame)
        lmList = detector.position(frame)
        print(lmList)

    ## To follow any body parts__Here is Elbow
    # lmList = detector.position(frame, draw=False)
    # if len(lmList) != 0:
    #     print(lmList[14])
    #     cv2.circle(frame, (lmList[14][1], lmList[14][2]), 15, (159, 19, 75), cv2.FILLED)

        current_time = time.time()
        fps = 1/ (current_time-pre_time)
        pre_time = current_time
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

        # Black Box
        start_point = (0, 0)
        end_point = (102, 32)
        color = (255, 255, 255)
        thickness = -1

        black_box = cv2.rectangle(frame, start_point, end_point, color, thickness) 

        cv2.putText(
            black_box,
            str(int(fps)),
            (31, 26),
            font, 1,
            (255, 200, 0),
            2
        )

        cv2.imshow("Camera", frame)
        if cv2.waitKey(10) == ord('0'):
            break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()    