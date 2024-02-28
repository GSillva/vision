import cv2
import math
import time
import argparse


import cv2
import math
import time

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

        self.cap = cv2.VideoCapture(0)
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
            if confidence > 0.8:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])

        return bboxes

    def run(self):
        while cv2.waitKey(1) < 0:
            t = time.time()
            hasFrame, frame = self.cap.read()

            if not hasFrame:
                cv2.waitKey()
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            bboxes = self.detect_face(small_frame)
            if not bboxes:
                print("No face Detected, Checking next frame")
                continue

            for bbox in bboxes:
                face = small_frame[max(0, bbox[1] - self.padding):min(bbox[3] + self.padding, frame.shape[0] - 1),
                       max(0, bbox[0] - self.padding):min(bbox[2] + self.padding, frame.shape[1] - 1)]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
                self.genderNet.setInput(blob)
                genderPreds = self.genderNet.forward()
                gender = self.genderList[genderPreds[0].argmax()]
                print("Gender: {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

                self.ageNet.setInput(blob)
                agePreds = self.ageNet.forward()
                age = self.ageList[agePreds[0].argmax()]
                print("Age Output: {}".format(agePreds))
                print("Age: {}, conf = {:.3f}".format(age, agePreds[0].max()))

                label = "{},{}".format(gender, age)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow("Age Gender Demo", frame)

            print("Time: {:.3f}".format(time.time() - t))

        self.cap.release()
        cv2.destroyAllWindows()

detector = AgeGenderDetector()
detector.run()
