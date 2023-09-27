import os
import random
import cv2
import numpy as np
import time

import tensorflow.lite as tflite


threshold = 0.7

img_dir = "/work/data/pet/images/"
model_path = "/work/data/pet/pet.tflite"
labels = [
    'Abyssinian',
    'american_bulldog',
    'american_pit_bull_terrier',
    'basset_hound',
    'beagle',
    'Bengal',
    'Birman',
    'Bombay',
    'boxer',
    'British_Shorthair',
    'chihuahua',
    'Egyptian_Mau',
    'english_cocker_spaniel',
    'english_setter',
    'german_shorthaired',
    'great_pyrenees',
    'havanese',
    'japanese_chin',
    'keeshond',
    'leonberger',
    'Maine_Coon',
    'miniature_pinscher',
    'newfoundland',
    'Persian',
    'pomeranian',
    'pug',
    'Ragdoll',
    'Russian_Blue',
    'saint_bernard',
    'samoyed',
    'scottish_terrier',
    'shiba_inu',
    'Siamese',
    'Sphynx',
    'staffordshire_bull_terrier',
    'wheaten_terrier',
    'yorkshire_terrier',
]
label_offset = 0

interpreter = tflite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_, input_height, input_width, _ = input_details[0]['shape']

filenames = os.listdir(img_dir)
random.shuffle(filenames)
for filename in filenames:
    start = time.time()

    img = cv2.imread(img_dir + filename)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_bgr, (input_height, input_width))
    img_normalized = img_resize / 128 - 1

    input_scale, input_zeropoint = input_details[0]['quantization']
    img_quantized = (img_normalized / input_scale) + input_zeropoint
    img_expand = np.expand_dims(img_quantized, 0)
    inputs = img_expand.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], inputs)

    interpreter.invoke()

    num_detections_index = 2
    detection_classes_index = 3
    detection_scores_index = 0
    detection_boxes_index = 1

    output_ = interpreter.get_tensor(output_details[num_detections_index]['index'])
    output_scale, output_zeropoint = output_details[num_detections_index]['quantization']
    num_detections = int((output_[0] - output_zeropoint) * output_scale)

    output_ = interpreter.get_tensor(output_details[detection_classes_index]['index'])
    output_scale, output_zeropoint = output_details[detection_classes_index]['quantization']
    detection_classes = (output_[0] - output_zeropoint) * output_scale

    output_ = interpreter.get_tensor(output_details[detection_boxes_index]['index'])
    output_scale, output_zeropoint = output_details[detection_boxes_index]['quantization']
    detection_boxes = (output_[0] - output_zeropoint) * output_scale

    output_ = interpreter.get_tensor(output_details[detection_scores_index]['index'])
    output_scale, output_zeropoint = output_details[detection_scores_index]['quantization']
    detection_scores = (output_[0] - output_zeropoint) * output_scale

    for i in range(num_detections):
        if detection_scores[i] < threshold:
            continue
        label = labels[round(detection_classes[i]) - label_offset]

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
