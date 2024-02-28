import cv2
import numpy as np
import time
import pyrealsense2 as rs
import argparse
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


import cv2
import numpy as np
import pyrealsense2 as rs

class AgeGenderDetector:
    def __init__(self):
        self.faceProto = "./sex_age_recog.1/opencv_face_detector.pbtxt"
        self.faceModel = "./sex_age_recog.1/opencv_face_detector_uint8.pb"

        self.ageProto = "./sex_age_recog.1/age_deploy.prototxt"
        self.ageModel = "./sex_age_recog.1/age_net.caffemodel"

        self.genderProto = "./sex_age_recog.1/gender_deploy.prototxt"
        self.genderModel = "./sex_age_recog.1/gender_net.caffemodel"

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genderList = ['Male', 'Female']

        # Load the network
        self.ageNet = cv2.dnn.readNet(self.ageModel, self.ageProto)
        self.genderNet = cv2.dnn.readNet(self.genderModel, self.genderProto)
        self.faceNet = cv2.dnn.readNet(self.faceModel, self.faceProto)

        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)

        self.padding = 20

    def detect_face(self, frame):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                      [104, 117, 123], True, False)

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        bboxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.75:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])

        return bboxes

    def run(self):
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert RealSense frame to OpenCV format
            frame = np.asanyarray(color_frame.get_data())

            bboxes = self.detect_face(frame)
            if not bboxes:
                print("Nenhum rosto detectado, verificando próximo frame")
                continue

            for bbox in bboxes:
                face = frame[max(0, bbox[1] - self.padding):min(bbox[3] + self.padding, frame.shape[0] - 1),
                       max(0, bbox[0] - self.padding):min(bbox[2] + self.padding, frame.shape[1] - 1)]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
                self.genderNet.setInput(blob)
                genderPreds = self.genderNet.forward()
                gender = self.genderList[genderPreds[0].argmax()]
                print("Gênero: {}, confiança = {:.3f}".format(gender, genderPreds[0].max()))

                self.ageNet.setInput(blob)
                agePreds = self.ageNet.forward()
                age = self.ageList[agePreds[0].argmax()]
                print("Saída de idade: {}".format(agePreds))
                print("Idade: {}, confiança = {:.3f}".format(age, agePreds[0].max()))

                label = "{},{}".format(gender, age)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow("Demo de Idade e Gênero", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close OpenCV windows
        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = AgeGenderDetector()
    detector.run()
