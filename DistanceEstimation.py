import cv2 as cv
import numpy as np
import time

# Distance constants
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES

# Object detector constants
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for detected objects
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
FONTS = cv.FONT_HERSHEY_COMPLEX

# Load class names from 'classes.txt'
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Set up YOLO model
yoloNet = cv.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(320, 320), scale=1 / 255, swapRB=True)


def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    data_list = []
    for classid, score, box in zip(classes, scores, boxes):
        if class_names[classid] == "person":  # Only consider "person" class
            label = "%s : %f" % (class_names[classid], score)
            cv.rectangle(image, box, GREEN, 2)
            cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, GREEN, 2)
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    return (width_in_rf * measured_distance) / real_width


def distance_finder(focal_length, real_width, width_in_frame):
    return (real_width * focal_length) / width_in_frame


ref_person = cv.imread("ReferenceImages/image14.png")
person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

# Initialize 4 camera captures
caps = [cv.VideoCapture(i) for i in range(5)]

# 원하는 프레임 크기
FRAME_WIDTH, FRAME_HEIGHT = 320, 240

while True:
    start_time = time.time()

    frames = []
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {i} is not accessible")
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)  # 빈 화면 대체
        else:
            frame = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))  # 프레임 크기 조정
            data = object_detector(frame)
            for d in data:
                if d[0] == "person":
                    distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
                    x, y = d[2]
                    cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
                    distance_cm = distance * 2.54
                    cv.putText(
                        frame,
                        f"Dis: {round(distance_cm, 2)} cm",
                        (x + 5, y + 13),
                        FONTS,
                        0.32,
                        GREEN,
                        2,
                    )
            elapsed_time = time.time() - start_time
            fps = 1 / elapsed_time if elapsed_time > 0 else 0
            cv.putText(
                frame,
                f"FPS: {round(fps, 2)}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        frames.append(frame)

    # 2x2 Grid 형태로 영상 합치기
    top_row = np.hstack((frames[1], frames[2]))  # 위쪽 두 영상
    bottom_row = np.hstack((frames[3], frames[4]))  # 아래쪽 두 영상
    combined_frame = np.vstack((top_row, bottom_row))  # 위아래로 결합

    cv.imshow("Combined View", combined_frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break

for cap in caps:
    cap.release()
cv.destroyAllWindows()
