import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetect = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetect.FaceDetection(self.minDetectionCon)
        self.mpDraw = mp.solutions.drawing_utils
         
    def findFace(self, frame, draw = True):     
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(frameRGB)

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxs = []
                bboxC = detection.location_data.relative_bounding_box
                frame_h, frame_w, frame_c = frame.shape
                bbox = int(bboxC.xmin * frame_w), int(bboxC.ymin * frame_h), \
                    int(bboxC.width * frame_w), int(bboxC.height * frame_h)
                bboxs.append([bbox, detection.score])

                if draw:    
                    frame = self.Design(frame, bbox)
                    cv2.putText(
                        frame,
                        f"{int(detection.score[0]*100)}%",
                        (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 255),
                        2
                    )
        return frame, bboxs
    
    def Design(self, frame, bbox, length=20, thickness=4, color = (0, 125, 255)):
        x, y, h, w  = bbox
        x1, y1 = x+w, y+h
        cv2.rectangle(frame, bbox, (255, 0, 255), 2)

        # Top Left (x, y)
        cv2.line(frame, (x,y), (x+length , y), color, thickness)
        cv2.line(frame, (x,y), (x , y+length), color, thickness)

        # Top Right (x1, y)
        cv2.line(frame, (x1,y), (x1-length , y), color, thickness)
        cv2.line(frame, (x1,y), (x1 , y+length), color, thickness)

        # Bottom Left (x, y1)
        cv2.line(frame, (x,y1), (x+length , y1), color, thickness)
        cv2.line(frame, (x,y1), (x , y1-length), color, thickness)

        # Top Right (x1, y1)
        cv2.line(frame, (x1,y1), (x1-length , y1), color, thickness)
        cv2.line(frame, (x1,y1), (x1 , y1-length), color, thickness)

        return frame

def main():
    # vdo = 'v00.mp4'
    cap = cv2.VideoCapture(0)
    pre_time = 0
    current_time = 0
    detector = FaceDetector()

    while True:
        try:
            succes, frame = cap.read()
            frame, bboxs = detector.findFace(frame)
            current_time = time.time()
            fps = 1/ (current_time-pre_time)
            pre_time = current_time

            # FPS Box
            start_point = (0, 0)
            end_point = (110, 32)
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
                (127, 0, 255),
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
        except:
            continue    
        
    cap.release()
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()    