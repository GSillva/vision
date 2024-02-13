import cv2
import math
import time
import argparse
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def detecFace(net, frame, conf_threshold=0.75):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                 [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0),
                          int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, bboxes

def image_callback(msg):
    global bridge
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        print(e)
    else:
        frameFace, bboxes = detecFace(faceNet, cv_image)
        if not bboxes:
            print("No face Detected, Checking next frame")
            return
        for bbox in bboxes:
            face = cv_image[max(0, bbox[1] - padding):min(bbox[3] + padding, cv_image.shape[0] - 1),
                            max(0, bbox[0] - padding):min(bbox[2] + padding, cv_image.shape[1] - 1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print("Age Output : {}".format(agePreds))
            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            label = "{},{}".format(gender, age)
            cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Age Gender Demo", frameFace)
        cv2.waitKey(1)

def main(args=None):
    global faceNet, ageNet, genderNet, bridge

    rclpy.init(args=args)
    node = rclpy.create_node('age_gender_detector')

    bridge = CvBridge()

    faceProto = "./sex_age_recog.1/opencv_face_detector.pbtxt"
    faceModel = "./sex_age_recog.1/opencv_face_detector_uint8.pb"

    ageProto = "./sex_age_recog.1/age_deploy.prototxt"
    ageModel = "./sex_age_recog.1/age_net.caffemodel"

    genderProto = "./sex_age_recog.1/gender_deploy.prototxt"
    genderModel = "./sex_age_recog.1/gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load the network
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    cap = cv2.VideoCapture(0)
    padding = 20

    subscription = node.create_subscription(Image, 'camera/image', image_callback, 10)

    while rclpy.ok():
        rclpy.spin_once(node)

    cap.release()
    cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()


    main()
