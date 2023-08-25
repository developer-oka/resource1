import os
import cv2
import numpy as np
import time

try:
    import tflite_runtime.interpreter as tflite #module
except:
    import tensorflow.lite as tflite #Ubuntu


img_dir = "/work/data/pic_e/"
model_path = "/work/ide/workspace/LearnigTF1/models/research/model.tflite"

threshold = 0.7

# label info
labels = ['car']
label_offset = 0

# tensorflow lite initialize
interpreter = tflite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# get input size
_, inference_height, inference_width, _ = input_details[0]['shape']


def run_inference_for_single_image(input_img):
    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_bgr, (inference_width, inference_height))
    _input = np.expand_dims(img_resized, axis=0)

    # set input data
    interpreter.set_tensor(input_details[0]['index'], _input)

    # run inference
    interpreter.invoke()

    detections_num_index = 3
    classes_index = 1
    scores_index = 2
    boxes_index = 0

    # get output data
    _detections_num = int(interpreter.get_tensor(output_details[detections_num_index]['index'])[0])
    _classes = interpreter.get_tensor(output_details[classes_index]['index'])[0]
    _scores = interpreter.get_tensor(output_details[scores_index]['index'])[0]
    _boxes = interpreter.get_tensor(output_details[boxes_index]['index'])[0]

    return _detections_num, _classes, _scores, _boxes


# get input date (evaluation data)
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
        class_ = labels[int(classes[i]) + label_offset]

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
                    org=(x1, y2 + 10),
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
