import os
import time
import random
import numpy as np
import tensorflow as tf
import cv2

img_dir = "/work/data/pet/images/"
model_path = "/work/ide/workspace/ObjectDetection/models/research/test_out/saved_model"

threshold = 0.5

labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "12", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "26", "backpack", "umbrella", "29", "30",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "45", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "66", "dining table", "68", "69", "toilet",
    "71", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "83", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

label_offset = 1

detection_model = tf.saved_model.load(model_path)
model_fn = detection_model.signatures["serving_default"]


def run_inference_for_single_image(input_img):
    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    _input = np.expand_dims(img_bgr, axis=0)

    tensor = tf.convert_to_tensor(_input)

    inference_info = model_fn(tensor)

    _num_detections = int(inference_info['num_detections'][0])
    _detection_classes = inference_info['detection_classes'][0].numpy()
    _detection_scores = inference_info['detection_scores'][0].numpy()
    _detection_boxes = inference_info['detection_boxes'][0].numpy()

    return _num_detections, _detection_classes, _detection_scores, _detection_boxes


filenames = os.listdir(img_dir)
random.shuffle(filenames)
for filename in filenames:
    start = time.time()

    img = cv2.imread(img_dir + filename)

    num_detections, detection_classes, detection_scores, detection_boxes = run_inference_for_single_image(img)

    for i in range(num_detections):
        if detection_scores[i] < threshold:
            continue

        label = labels[int(detection_classes[i]) - label_offset]

        h, w, _ = img.shape
        box = detection_boxes[i] * np.array([h, w, h, w])
        y1, x1, y2, x2 = box.astype(int)

        cv2.rectangle(img=img,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=[0, 0, 255])

        score = label + ':%.1f%%' % (detection_scores[i] * 100.0)
        cv2.putText(img=img,
                    text=score,
                    org=(x1, y2 + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=[0, 0, 255])

    _time = (time.time() - start) * 1000
    cv2.putText(img=img,
                text='%dms' % _time,
                org=(10, 30),
                fontScale=1,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                thickness=1,
                color=[0, 0, 255])
    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)
