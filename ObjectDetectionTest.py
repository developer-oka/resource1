import os
import time
import numpy as np
import tensorflow as tf
import cv2

img_dir = "/work/data/VOCdevkit/VOC2012/JPEGImages/"
model_path = "/work/ide/workspace/LearnigTF1/models/research/test_out/saved_model"

threshold = 0.5

# label info
labels = [
"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
"fire hydrant", "12", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
"cow", "elephant", "bear", "zebra", "giraffe", "26", "backpack", "umbrella", "29", "30",
"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
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
    # _input = img_expand.astype("uint8")

    tensor = tf.convert_to_tensor(_input)

    inference_info = model_fn(tensor)

    # get output data
    _detections_num = int(inference_info['num_detections'][0])
    _classes = inference_info['detection_classes'][0].numpy()
    _scores = inference_info['detection_scores'][0].numpy()
    _boxes = inference_info['detection_boxes'][0].numpy()

    return _detections_num, _classes, _scores, _boxes


# tensorflow initialize


# get input date (valuation data)
filenames = os.listdir(img_dir)
for filename in filenames:
    start = time.time()

    image_path = img_dir + filename
    print(image_path)
    img = cv2.imread(image_path)

    detections_num, classes, scores, boxes = run_inference_for_single_image(img)

    for i in range(detections_num):
        if scores[i] < threshold:
            continue

        # class
        class_ = labels[int(classes[i]) - label_offset]

        h, w, _ = img.shape
        box = boxes[i] * np.array([h, w, h, w])
        y1, x1, y2, x2 = box.astype(int)

        # Draw object detection boxes
        cv2.rectangle(img=img,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=[0, 0, 255])

        # Draw class and score
        score = class_ + ':%.1f%%' % (scores[i] * 100.0)
        cv2.putText(img=img,
                    text=score,
                    org=(x1, y2 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=[0, 0, 255])

    end = time.time()
    fps = '%0.1ffps' % (1.0 / (end - start))
    cv2.putText(img=img,
                text=fps,
                org=(10, 50),
                fontScale=1,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                color=[0, 0, 255])
    cv2.imshow('Object Detection', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
